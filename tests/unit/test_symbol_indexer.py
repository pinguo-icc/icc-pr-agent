"""Unit tests for src/symbol_indexer.py."""

from src.models import SymbolEntry
from src.symbol_indexer import SymbolIndex, SymbolIndexer


def _entry(name="Foo", file_path="src/foo.go", kind="function", language="go"):
    return SymbolEntry(
        name=name, signature=f"func {name}()", file_path=file_path,
        line_number=1, kind=kind, language=language,
    )


class TestSymbolIndex:
    def test_lookup_found(self):
        idx = SymbolIndex([_entry("CreateOrder"), _entry("DeleteOrder")])
        results = idx.lookup("CreateOrder")
        assert len(results) == 1
        assert results[0].name == "CreateOrder"

    def test_lookup_not_found(self):
        idx = SymbolIndex([_entry("CreateOrder")])
        results = idx.lookup("NonExistent")
        assert results == []

    def test_lookup_with_file_hint(self):
        e1 = _entry("Foo", file_path="pkg/a.go")
        e2 = _entry("Foo", file_path="pkg/b.go")
        idx = SymbolIndex([e1, e2])
        results = idx.lookup("Foo", file_hint="b.go")
        assert len(results) == 1
        assert results[0].file_path == "pkg/b.go"

    def test_lookup_file_hint_no_match_returns_all(self):
        e1 = _entry("Foo", file_path="pkg/a.go")
        idx = SymbolIndex([e1])
        results = idx.lookup("Foo", file_hint="nonexistent.go")
        assert len(results) == 1  # Falls back to all matches

    def test_empty_index(self):
        idx = SymbolIndex([])
        assert idx.lookup("anything") == []
        assert idx.entries == []


class TestSymbolIndexSerialization:
    def test_round_trip(self):
        entries = [_entry("A"), _entry("B", file_path="b.py", language="python")]
        idx = SymbolIndex(entries)
        data = idx.to_cache_dict()
        restored = SymbolIndex.from_cache_dict(data)
        assert len(restored.entries) == 2
        assert restored.entries[0].name == "A"
        assert restored.entries[1].language == "python"

    def test_from_empty_cache(self):
        idx = SymbolIndex.from_cache_dict({})
        assert idx.entries == []

    def test_from_cache_with_entries(self):
        data = {"entries": [
            {"name": "X", "signature": "func X()", "file_path": "x.go",
             "line_number": 5, "kind": "function", "language": "go"},
        ]}
        idx = SymbolIndex.from_cache_dict(data)
        assert len(idx.entries) == 1
        assert idx.entries[0].name == "X"


class TestSymbolIndexerExcludedDirs:
    def test_excluded_dirs_detected(self):
        indexer = SymbolIndexer(cache_dir="/tmp")
        assert indexer._is_excluded_dir("vendor")
        assert indexer._is_excluded_dir("vendor/pkg")
        assert indexer._is_excluded_dir("node_modules")
        assert indexer._is_excluded_dir("node_modules/foo")
        assert indexer._is_excluded_dir("third_party")
        assert indexer._is_excluded_dir(".git")
        assert not indexer._is_excluded_dir("src")
        assert not indexer._is_excluded_dir("internal/vendor_utils")

    def test_dot_not_excluded(self):
        indexer = SymbolIndexer(cache_dir="/tmp")
        assert not indexer._is_excluded_dir(".")


class TestSymbolIndexerLanguageDetection:
    def test_go(self):
        assert SymbolIndexer._detect_language("main.go") == "go"

    def test_proto(self):
        assert SymbolIndexer._detect_language("user.proto") == "proto"

    def test_typescript(self):
        assert SymbolIndexer._detect_language("app.ts") == "typescript"
        assert SymbolIndexer._detect_language("app.tsx") == "typescript"

    def test_javascript(self):
        assert SymbolIndexer._detect_language("app.js") == "javascript"

    def test_python(self):
        assert SymbolIndexer._detect_language("main.py") == "python"

    def test_unknown(self):
        assert SymbolIndexer._detect_language("readme.md") is None
        assert SymbolIndexer._detect_language("data.json") is None


class TestSymbolIndexerCacheFailure:
    def test_load_cache_returns_none_on_missing_file(self):
        result = SymbolIndexer._load_cache("/nonexistent/path.json")
        assert result is None

    def test_load_cache_returns_none_on_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        result = SymbolIndexer._load_cache(str(bad_file))
        assert result is None
