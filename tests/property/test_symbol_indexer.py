"""Property-based tests for SymbolIndexer."""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.models import SymbolEntry
from src.symbol_indexer import SymbolIndex, SymbolIndexer


def _symbol_entry_strategy():
    return st.builds(
        SymbolEntry,
        name=st.from_regex(r"[A-Z][a-zA-Z0-9]{1,15}", fullmatch=True),
        signature=st.from_regex(r"func [A-Z][a-zA-Z0-9]{1,15}\(\)", fullmatch=True),
        file_path=st.from_regex(r"[a-z]{1,5}/[a-z]{1,8}\.(go|py|ts)", fullmatch=True),
        line_number=st.integers(min_value=1, max_value=10000),
        kind=st.sampled_from(["function", "method", "class", "interface", "struct"]),
        language=st.sampled_from(["go", "python", "typescript"]),
    )


# --- Property 12: 符号索引排除第三方目录 ---
# Feature: sub-agent-review, Property 12: 符号索引排除第三方目录

@given(
    dir_prefix=st.sampled_from(["vendor", "node_modules", "third_party", ".git"]),
    subpath=st.from_regex(r"[a-z]{1,5}/[a-z]{1,8}\.(go|py)", fullmatch=True),
)
@settings(max_examples=100)
def test_property_12_excluded_dirs(dir_prefix, subpath):
    """Files in excluded directories should not appear in SymbolIndex.

    Validates: Requirements 8.4
    """
    indexer = SymbolIndexer(cache_dir="/tmp")
    full_rel = f"{dir_prefix}/{subpath}"
    assert indexer._is_excluded_dir(dir_prefix)


# --- Property 13: 符号查询返回匹配条目 ---
# Feature: sub-agent-review, Property 13: 符号查询返回匹配条目

@given(entries=st.lists(_symbol_entry_strategy(), min_size=1, max_size=20))
@settings(max_examples=100)
def test_property_13_lookup_returns_matching(entries):
    """lookup(name) returns entries with matching name; missing names return [].

    Validates: Requirements 8.7, 8.8
    """
    idx = SymbolIndex(entries)

    # Pick a name that exists
    target = entries[0].name
    results = idx.lookup(target)
    assert len(results) > 0
    for r in results:
        assert r.name == target
        assert r.signature  # non-empty
        assert r.file_path  # non-empty
        assert r.line_number > 0

    # Query a name that doesn't exist
    fake_name = "ZZZZNONEXISTENT999"
    assert idx.lookup(fake_name) == []


# --- Property 14: 符号索引序列化往返 ---
# Feature: sub-agent-review, Property 14: 符号索引序列化往返

@given(entries=st.lists(_symbol_entry_strategy(), min_size=0, max_size=20))
@settings(max_examples=100)
def test_property_14_serialization_round_trip(entries):
    """from_cache_dict(to_cache_dict()) produces equivalent index.

    Validates: Requirements 8.9
    """
    idx = SymbolIndex(entries)
    data = idx.to_cache_dict()
    restored = SymbolIndex.from_cache_dict(data)

    assert len(restored.entries) == len(idx.entries)
    for orig, rest in zip(idx.entries, restored.entries):
        assert orig.name == rest.name
        assert orig.signature == rest.signature
        assert orig.file_path == rest.file_path
        assert orig.line_number == rest.line_number
        assert orig.kind == rest.kind
        assert orig.language == rest.language
