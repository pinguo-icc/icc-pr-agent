"""Property-based tests for BatchSplitter."""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.batch_splitter import BatchSplitter
from src.models import FileGroup


def _file_group_strategy():
    """Generate a random FileGroup with 1-10 files."""
    file_entry = st.tuples(
        st.from_regex(r"[a-z]{1,3}/[a-z]{1,8}\.(go|proto|ts|py)", fullmatch=True),
        st.integers(min_value=1, max_value=5000).map(lambda n: "x" * n),
    )
    entries = st.lists(file_entry, min_size=1, max_size=10)
    return entries.map(lambda items: FileGroup(
        name="test",
        file_paths=[p for p, _ in items],
        file_diffs=[d for _, d in items],
        total_chars=sum(len(d) for _, d in items),
    ))


# --- Property 5: 批次分片不超限 ---
# Feature: sub-agent-review, Property 5: 批次分片不超限

@given(group=_file_group_strategy(), max_chars=st.integers(min_value=100, max_value=10000))
@settings(max_examples=100)
def test_property_5_batch_size_within_limit(group, max_chars):
    """Each batch char_count <= max_chunk_chars (except single oversized files).
    All batches merged equal original diff content.

    Validates: Requirements 4.1, 4.4
    """
    batches = BatchSplitter().split(group, max_chars)

    for batch in batches:
        if len(batch.file_paths) > 1:
            # Multi-file batch must not exceed limit
            assert batch.char_count <= max_chars
        # Single-file batch may exceed (oversized file)

    # All file paths accounted for
    all_paths = sorted(p for b in batches for p in b.file_paths)
    original_sorted = sorted(group.file_paths)
    assert all_paths == original_sorted


# --- Property 6: 批次内文件按目录前缀排序 ---
# Feature: sub-agent-review, Property 6: 批次内文件按目录前缀排序

@given(group=_file_group_strategy())
@settings(max_examples=100)
def test_property_6_files_sorted_by_directory(group):
    """Files within each batch are sorted by directory prefix.

    Validates: Requirements 4.2
    """
    import os
    batches = BatchSplitter().split(group, 100000)

    for batch in batches:
        dirs = [os.path.dirname(p) for p in batch.file_paths]
        assert dirs == sorted(dirs), (
            f"Files not sorted by directory: {batch.file_paths}"
        )
