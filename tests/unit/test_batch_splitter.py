"""Unit tests for src/batch_splitter.py."""

from src.batch_splitter import BatchSplitter
from src.models import FileGroup


def _fg(name: str, paths: list[str], diffs: list[str]) -> FileGroup:
    return FileGroup(name=name, file_paths=paths, file_diffs=diffs,
                     total_chars=sum(len(d) for d in diffs))


class TestBatchSplitterBasic:
    def test_empty_group(self):
        fg = _fg("empty", [], [])
        batches = BatchSplitter().split(fg, 1000)
        assert batches == []

    def test_single_file_within_limit(self):
        fg = _fg("backend", ["main.go"], ["x" * 100])
        batches = BatchSplitter().split(fg, 1000)
        assert len(batches) == 1
        assert batches[0].file_paths == ["main.go"]
        assert batches[0].char_count == 100

    def test_multiple_files_within_limit(self):
        fg = _fg("backend", ["a.go", "b.go"], ["x" * 50, "y" * 50])
        batches = BatchSplitter().split(fg, 1000)
        assert len(batches) == 1
        assert len(batches[0].file_paths) == 2


class TestBatchSplitterOverflow:
    def test_single_file_exceeds_limit(self):
        """Requirement 4.4: single oversized file gets its own batch."""
        fg = _fg("backend", ["big.go"], ["x" * 5000])
        batches = BatchSplitter().split(fg, 1000)
        assert len(batches) == 1
        assert batches[0].file_paths == ["big.go"]
        assert batches[0].char_count == 5000

    def test_split_into_multiple_batches(self):
        fg = _fg("backend",
                 ["a.go", "b.go", "c.go"],
                 ["x" * 400, "y" * 400, "z" * 400])
        batches = BatchSplitter().split(fg, 500)
        assert len(batches) >= 2
        # All files accounted for
        all_paths = [p for b in batches for p in b.file_paths]
        assert sorted(all_paths) == ["a.go", "b.go", "c.go"]


class TestBatchSplitterSorting:
    def test_directory_prefix_sorting(self):
        """Requirement 4.2: files sorted by directory prefix."""
        fg = _fg("backend",
                 ["z/file.go", "a/file.go", "m/file.go"],
                 ["x" * 10, "y" * 10, "z" * 10])
        batches = BatchSplitter().split(fg, 10000)
        assert len(batches) == 1
        assert batches[0].file_paths == ["a/file.go", "m/file.go", "z/file.go"]

    def test_related_files_same_directory(self):
        """Requirement 4.3: related files in same dir stay together."""
        fg = _fg("backend",
                 ["api/user.proto", "api/user_service.go", "pkg/other.go"],
                 ["x" * 100, "y" * 100, "z" * 100])
        batches = BatchSplitter().split(fg, 250)
        # api/ files should be in the same batch
        api_batch = [b for b in batches if "api/user.proto" in b.file_paths]
        assert len(api_batch) == 1
        assert "api/user_service.go" in api_batch[0].file_paths


class TestBatchSplitterContent:
    def test_diff_content_preserved(self):
        """All batch diffs concatenated equal original diffs (sorted)."""
        diffs = ["diff_a\n", "diff_b\n", "diff_c\n"]
        fg = _fg("backend", ["a.go", "b.go", "c.go"], diffs)
        batches = BatchSplitter().split(fg, 10000)
        merged = "".join(b.diff_content for b in batches)
        # Since files are sorted and all in same dir, order preserved
        assert merged == "".join(diffs)

    def test_batch_indices_sequential(self):
        fg = _fg("backend",
                 ["a.go", "b.go", "c.go"],
                 ["x" * 400, "y" * 400, "z" * 400])
        batches = BatchSplitter().split(fg, 500)
        for i, b in enumerate(batches):
            assert b.batch_index == i
            assert b.group_name == "backend"
