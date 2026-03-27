"""File grouper that assigns diff files to domain groups."""

from __future__ import annotations

import fnmatch
import logging
import re

from src.models import FileGroup

logger = logging.getLogger(__name__)


class FileGrouper:
    """按关联域对 diff 文件进行分组。"""

    DEFAULT_FILE_GROUPS: dict[str, list[str]] = {
        "backend": ["*.go", "*.proto"],
        "frontend": [
            "*.ts", "*.tsx", "*.js", "*.jsx", "*.vue",
            "*.css", "*.less", "*.scss",
        ],
        "infra": ["Dockerfile*", "Makefile", "*.yml"],
        "default": ["*"],
    }

    _DIFF_HEADER_RE = re.compile(r"^diff --git a/.+ b/(.+)$")

    def __init__(self, file_groups: dict[str, list[str]] | None = None) -> None:
        self._file_groups = file_groups if file_groups is not None else self.DEFAULT_FILE_GROUPS

    def group(self, diff: str) -> dict[str, FileGroup]:
        """将 unified diff 按文件分组。返回 {group_name: FileGroup}，跳过空组。"""
        # Split diff into per-file sections
        file_diffs: dict[str, str] = {}
        current_path: str | None = None
        current_lines: list[str] = []

        for line in diff.splitlines(keepends=True):
            stripped = line.rstrip("\n\r")
            path = self._extract_file_path(stripped)
            if path is not None:
                # Save previous file
                if current_path is not None:
                    file_diffs[current_path] = "".join(current_lines)
                current_path = path
                current_lines = [line]
            else:
                current_lines.append(line)

        # Save last file
        if current_path is not None:
            file_diffs[current_path] = "".join(current_lines)

        # Assign files to groups
        groups: dict[str, list[tuple[str, str]]] = {}  # group_name -> [(path, diff)]
        for path, fdiff in file_diffs.items():
            gname = self._match_group(path, self._file_groups)
            groups.setdefault(gname, []).append((path, fdiff))

        # Build FileGroup objects, skip empty
        result: dict[str, FileGroup] = {}
        for gname, items in groups.items():
            paths = [p for p, _ in items]
            diffs = [d for _, d in items]
            total = sum(len(d) for d in diffs)
            result[gname] = FileGroup(name=gname, file_paths=paths, file_diffs=diffs, total_chars=total)

        return result

    @staticmethod
    def _extract_file_path(diff_header_line: str) -> str | None:
        """从 'diff --git a/... b/...' 头部提取文件路径。"""
        m = FileGrouper._DIFF_HEADER_RE.match(diff_header_line)
        return m.group(1) if m else None

    @staticmethod
    def _match_group(file_path: str, file_groups: dict[str, list[str]]) -> str:
        """返回文件匹配的第一个分组名。无匹配时返回 'default'。"""
        basename = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
        for group_name, patterns in file_groups.items():
            for pattern in patterns:
                try:
                    if fnmatch.fnmatch(basename, pattern):
                        return group_name
                except Exception:
                    logger.warning("Invalid glob pattern '%s' in group '%s', skipping", pattern, group_name)
        return "default"
