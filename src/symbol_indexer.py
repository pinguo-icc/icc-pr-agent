"""Symbol indexer using tree-sitter for repository code analysis."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from src.exceptions import SymbolIndexError
from src.models import SymbolEntry

logger = logging.getLogger(__name__)


# Language extension mapping
_LANG_MAP: dict[str, str] = {
    ".go": "go",
    ".proto": "proto",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".py": "python",
}


class SymbolIndex:
    """符号索引，支持按名称查询。"""

    def __init__(self, entries: list[SymbolEntry]) -> None:
        self._entries = entries
        # Build name -> entries lookup
        self._by_name: dict[str, list[SymbolEntry]] = {}
        for e in entries:
            self._by_name.setdefault(e.name, []).append(e)

    @property
    def entries(self) -> list[SymbolEntry]:
        return self._entries

    def lookup(self, name: str, file_hint: str | None = None) -> list[SymbolEntry]:
        """按符号名称查询，可选 file_hint 缩小范围。"""
        candidates = self._by_name.get(name, [])
        if file_hint and candidates:
            filtered = [e for e in candidates if file_hint in e.file_path]
            return filtered if filtered else candidates
        return candidates

    def to_cache_dict(self) -> dict:
        """序列化为可 JSON 持久化的字典。"""
        return {
            "entries": [asdict(e) for e in self._entries],
        }

    @classmethod
    def from_cache_dict(cls, data: dict) -> SymbolIndex:
        """从缓存字典反序列化。"""
        entries = [
            SymbolEntry(**e) for e in data.get("entries", [])
        ]
        return cls(entries)


class SymbolIndexer:
    """仓库符号索引构建器。"""

    EXCLUDED_DIRS: list[str] = ["vendor/", "node_modules/", "third_party/", ".git/"]

    def __init__(
        self,
        cache_dir: str,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._exclude_patterns = exclude_patterns or []

    def build(
        self,
        repo_url: str,
        branch: str,
        repo_dir: str | None = None,
        changed_files: list[str] | None = None,
    ) -> SymbolIndex:
        """构建或增量更新符号索引。"""
        # Resolve persistent repo directory
        if not repo_dir:
            repo_dir = self._ensure_repo(repo_url, branch)

        # Try loading cache
        cache_path = self._cache_path(repo_url)
        cached_index = self._load_cache(cache_path)

        if cached_index and cached_index.entries and changed_files:
            # Incremental update: re-parse only changed files
            logger.info("Incremental symbol index update for %d files", len(changed_files))
            existing = {e.file_path: e for e in cached_index.entries}
            # Remove old entries for changed files
            for f in changed_files:
                existing.pop(f, None)
            # Re-parse changed files
            for f in changed_files:
                full_path = os.path.join(repo_dir, f)
                lang = self._detect_language(f)
                if lang and os.path.isfile(full_path):
                    new_entries = self._parse_file(full_path, lang, f)
                    for e in new_entries:
                        existing[e.file_path + ":" + e.name] = e
            entries = list(existing.values())
            index = SymbolIndex(entries)
            self._save_cache(cache_path, index)
            return index

        # Full build
        entries = self._scan_directory(repo_dir)
        index = SymbolIndex(entries)
        self._save_cache(cache_path, index)
        return index

    def _repo_dir_path(self, repo_url: str) -> str:
        """Generate persistent repo directory path: {cache_dir}/{platform}/{owner}/{repo}/code/."""
        platform, owner, repo = self._parse_repo_url(repo_url)
        return os.path.join(self._cache_dir, platform, owner, repo, "code")

    @staticmethod
    def _parse_repo_url(repo_url: str) -> tuple[str, str, str]:
        """Parse repo URL into (platform, owner, repo).

        Example: https://github.com/pinguo-icc/order-svc.git -> (github, pinguo-icc, order-svc)
        """
        import re
        m = re.match(r"https?://([^/]+)/([^/]+)/([^/]+?)(?:\.git)?$", repo_url)
        if m:
            host = m.group(1)  # github.com
            owner = m.group(2)
            repo = m.group(3)
            # Normalize host to platform name
            if "github" in host:
                platform = "github"
            elif "gitlab" in host:
                platform = "gitlab"
            else:
                platform = host.replace(".", "_")
            return platform, owner, repo
        # Fallback: use sanitized URL
        safe = repo_url.replace("/", "_").replace(":", "_").replace(".", "_")
        return "unknown", safe, "repo"

    def _ensure_repo(self, repo_url: str, branch: str) -> str:
        """Clone or update repo in persistent directory under cache_dir/repos/."""
        repo_dir = self._repo_dir_path(repo_url)
        git_dir = os.path.join(repo_dir, ".git")

        if os.path.isdir(git_dir):
            # Already cloned — fetch and reset to latest
            logger.info("更新已有仓库: %s (branch=%s)", repo_dir, branch)
            try:
                subprocess.run(
                    ["git", "fetch", "origin", branch, "--depth=1"],
                    cwd=repo_dir, check=True, capture_output=True, timeout=120,
                )
                subprocess.run(
                    ["git", "reset", "--hard", f"origin/{branch}"],
                    cwd=repo_dir, check=True, capture_output=True, timeout=30,
                )
                subprocess.run(
                    ["git", "clean", "-fdx"],
                    cwd=repo_dir, check=True, capture_output=True, timeout=30,
                )
                return repo_dir
            except Exception as exc:
                logger.warning("仓库更新失败，将重新 clone: %s", exc)
                import shutil
                shutil.rmtree(repo_dir, ignore_errors=True)

        # Fresh clone
        logger.info("Clone 仓库: %s -> %s", repo_url, repo_dir)
        os.makedirs(repo_dir, exist_ok=True)
        try:
            cmd = ["git", "clone", "--depth=1", "--branch", branch, repo_url, repo_dir]
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            return repo_dir
        except Exception as exc:
            logger.warning("Failed to clone repo: %s", exc)
            raise SymbolIndexError(f"Clone failed: {exc}") from exc

    def _scan_directory(self, root: str) -> list[SymbolEntry]:
        """Scan all files in directory, excluding third-party dirs."""
        entries: list[SymbolEntry] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Filter out excluded directories
            rel_dir = os.path.relpath(dirpath, root)
            if self._is_excluded_dir(rel_dir):
                dirnames.clear()
                continue
            # Remove excluded dirs from traversal
            dirnames[:] = [
                d for d in dirnames
                if not self._is_excluded_dir(os.path.join(rel_dir, d))
            ]
            for fname in filenames:
                lang = self._detect_language(fname)
                if not lang:
                    continue
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root)
                try:
                    new_entries = self._parse_file(full_path, lang, rel_path)
                    entries.extend(new_entries)
                except Exception:
                    logger.warning("Failed to parse %s, skipping", rel_path)
        return entries

    def _is_excluded_dir(self, rel_path: str) -> bool:
        """Check if a relative path is in an excluded directory."""
        normalized = rel_path.replace("\\", "/")
        if normalized == ".":
            return False
        for excluded in self.EXCLUDED_DIRS:
            clean = excluded.rstrip("/")
            if normalized == clean or normalized.startswith(clean + "/"):
                return True
        return False

    def _parse_file(
        self, full_path: str, language: str, rel_path: str,
    ) -> list[SymbolEntry]:
        """Parse a single file using tree-sitter."""
        try:
            from tree_sitter_languages import get_parser
        except ImportError:
            logger.warning("tree-sitter-languages not installed, skipping parse")
            return []

        ts_lang = language
        if ts_lang == "proto":
            # tree-sitter-languages may not support proto
            return self._parse_proto_fallback(full_path, rel_path)

        try:
            parser = get_parser(ts_lang)
        except Exception:
            logger.warning("No tree-sitter parser for %s", ts_lang)
            return []

        try:
            source = Path(full_path).read_bytes()
            tree = parser.parse(source)
        except Exception:
            logger.warning("Failed to parse %s", rel_path)
            return []

        entries: list[SymbolEntry] = []
        self._walk_tree(tree.root_node, source, language, rel_path, entries)
        return entries

    def _walk_tree(
        self, node, source: bytes, language: str,
        rel_path: str, entries: list[SymbolEntry],
    ) -> None:
        """Walk AST and extract symbol entries."""
        kind = self._node_to_kind(node.type, language)
        if kind:
            name = self._extract_name(node, source, language)
            if name:
                line_num = node.start_point[0] + 1
                sig = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
                # Truncate long signatures
                first_line = sig.split("\n")[0]
                if len(first_line) > 200:
                    first_line = first_line[:200] + "..."
                entries.append(SymbolEntry(
                    name=name,
                    signature=first_line,
                    file_path=rel_path,
                    line_number=line_num,
                    kind=kind,
                    language=language,
                ))

        for child in node.children:
            self._walk_tree(child, source, language, rel_path, entries)

    @staticmethod
    def _node_to_kind(node_type: str, language: str) -> str | None:
        """Map tree-sitter node type to our kind enum."""
        mapping: dict[str, dict[str, str]] = {
            "go": {
                "function_declaration": "function",
                "method_declaration": "method",
                "type_spec": "struct",
            },
            "python": {
                "function_definition": "function",
                "class_definition": "class",
            },
            "typescript": {
                "function_declaration": "function",
                "class_declaration": "class",
                "interface_declaration": "interface",
                "export_statement": None,  # handled separately
            },
            "javascript": {
                "function_declaration": "function",
                "class_declaration": "class",
            },
        }
        lang_map = mapping.get(language, {})
        return lang_map.get(node_type)

    @staticmethod
    def _extract_name(node, source: bytes, language: str) -> str | None:
        """Extract symbol name from AST node."""
        # Go method_declaration: method name is the field_identifier after receiver
        if language == "go" and node.type == "method_declaration":
            for child in node.children:
                if child.type == "field_identifier":
                    return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            return None

        # Go type_spec: name is the type_identifier child
        if language == "go" and node.type == "type_spec":
            for child in node.children:
                if child.type == "type_identifier":
                    return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            return None

        # Default: first identifier/name child
        for child in node.children:
            if child.type in ("identifier", "name", "type_identifier"):
                return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        return None

    def _parse_proto_fallback(
        self, full_path: str, rel_path: str,
    ) -> list[SymbolEntry]:
        """Simple regex-based proto file parsing."""
        import re
        entries: list[SymbolEntry] = []
        try:
            content = Path(full_path).read_text(encoding="utf-8")
        except Exception:
            return entries

        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            # service Foo {
            m = re.match(r"service\s+(\w+)", stripped)
            if m:
                entries.append(SymbolEntry(
                    name=m.group(1), signature=stripped,
                    file_path=rel_path, line_number=i,
                    kind="service", language="proto",
                ))
            # rpc Foo(...)
            m = re.match(r"rpc\s+(\w+)", stripped)
            if m:
                entries.append(SymbolEntry(
                    name=m.group(1), signature=stripped,
                    file_path=rel_path, line_number=i,
                    kind="rpc", language="proto",
                ))
            # message Foo {
            m = re.match(r"message\s+(\w+)", stripped)
            if m:
                entries.append(SymbolEntry(
                    name=m.group(1), signature=stripped,
                    file_path=rel_path, line_number=i,
                    kind="message", language="proto",
                ))
        return entries

    @staticmethod
    def _detect_language(file_path: str) -> str | None:
        """根据文件扩展名检测语言。"""
        ext = os.path.splitext(file_path)[1].lower()
        return _LANG_MAP.get(ext)

    def _cache_path(self, repo_url: str) -> str:
        """Generate cache file path: {cache_dir}/{platform}/{owner}/{repo}/data/symbols.json."""
        platform, owner, repo = self._parse_repo_url(repo_url)
        cache_dir = os.path.join(self._cache_dir, platform, owner, repo, "data")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "symbols.json")

    @staticmethod
    def _load_cache(cache_path: str) -> SymbolIndex | None:
        """Load cached symbol index."""
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return SymbolIndex.from_cache_dict(data)
        except Exception:
            return None

    @staticmethod
    def _save_cache(cache_path: str, index: SymbolIndex) -> None:
        """Save symbol index to cache."""
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(index.to_cache_dict(), f, ensure_ascii=False)
        except Exception:
            logger.warning("Failed to save symbol cache to %s", cache_path)
