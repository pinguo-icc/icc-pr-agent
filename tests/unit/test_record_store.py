"""Unit tests for RecordStore."""

from __future__ import annotations

import json

from src.models import (
    ReviewDiffReport,
    ReviewIssue,
    ReviewRecord,
    ReviewResult,
)
from src.record_store import RecordStore


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_record(
    *,
    record_id: str = "rec-1",
    pr_id: str = "owner/repo#42",
    pr_url: str = "https://github.com/owner/repo/pull/42",
    platform: str = "github",
    version_id: str = "abc123",
    created_at: str = "2024-01-15T10:30:00",
    summary: str = "Looks good",
    diff_report: ReviewDiffReport | None = None,
) -> ReviewRecord:
    result = ReviewResult(
        summary=summary,
        issues=[
            ReviewIssue(
                file_path="src/main.py",
                line_number=10,
                severity="warning",
                category="quality",
                description="Unused import",
                suggestion="Remove it",
            )
        ],
        reviewed_at=created_at,
    )
    return ReviewRecord(
        record_id=record_id,
        pr_id=pr_id,
        pr_url=pr_url,
        platform=platform,
        version_id=version_id,
        review_result=result,
        diff_report=diff_report,
        created_at=created_at,
    )


# -------------------------------------------------------------------
# Tests — save
# -------------------------------------------------------------------

class TestRecordStoreSave:
    """Tests for RecordStore.save."""

    def test_save_creates_json_file(self, tmp_path):
        store = RecordStore(storage_dir=str(tmp_path))
        record = _make_record()
        store.save(record)

        expected_dir = tmp_path / "github" / "owner" / "repo" / "prs" / "42"
        assert expected_dir.is_dir()

        json_files = list(expected_dir.glob("*.json"))
        assert len(json_files) == 1

        with open(json_files[0], "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["record_id"] == "rec-1"
        assert data["pr_id"] == "owner/repo#42"

    def test_save_sanitizes_colons_in_filename(self, tmp_path):
        store = RecordStore(storage_dir=str(tmp_path))
        record = _make_record(
            created_at="2024-01-15T10:30:00",
            version_id="abc123",
        )
        store.save(record)

        pr_dir = tmp_path / "github" / "owner" / "repo" / "prs" / "42"
        json_files = list(pr_dir.glob("*.json"))
        assert len(json_files) == 1
        assert json_files[0].name == (
            "2024-01-15T10-30-00_abc123.json"
        )

    def test_save_multiple_records(self, tmp_path):
        store = RecordStore(storage_dir=str(tmp_path))
        r1 = _make_record(
            record_id="r1",
            created_at="2024-01-15T10:00:00",
            version_id="v1",
        )
        r2 = _make_record(
            record_id="r2",
            created_at="2024-01-16T14:00:00",
            version_id="v2",
        )
        store.save(r1)
        store.save(r2)

        pr_dir = tmp_path / "github" / "owner" / "repo" / "prs" / "42"
        json_files = list(pr_dir.glob("*.json"))
        assert len(json_files) == 2

    def test_save_with_diff_report(self, tmp_path):
        store = RecordStore(storage_dir=str(tmp_path))
        diff_report = ReviewDiffReport(
            improved=[{"issue": "fixed"}],
            unresolved=[],
            new_issues=[],
        )
        record = _make_record(diff_report=diff_report)
        store.save(record)

        pr_dir = tmp_path / "github" / "owner" / "repo" / "prs" / "42"
        json_files = list(pr_dir.glob("*.json"))
        with open(json_files[0], "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["diff_report"]["improved"] == [
            {"issue": "fixed"},
        ]


# -------------------------------------------------------------------
# Tests — get_latest
# -------------------------------------------------------------------

class TestRecordStoreGetLatest:
    """Tests for RecordStore.get_latest."""

    def test_returns_none_when_no_records(self, tmp_path):
        store = RecordStore(storage_dir=str(tmp_path))
        assert store.get_latest("owner/repo#42") is None

    def test_returns_none_when_dir_missing(self, tmp_path):
        store = RecordStore(
            storage_dir=str(tmp_path / "nonexistent"),
        )
        assert store.get_latest("owner/repo#42") is None

    def test_returns_most_recent_record(self, tmp_path):
        store = RecordStore(storage_dir=str(tmp_path))
        r1 = _make_record(
            record_id="r1",
            created_at="2024-01-15T10:00:00",
            version_id="v1",
        )
        r2 = _make_record(
            record_id="r2",
            created_at="2024-01-16T14:00:00",
            version_id="v2",
        )
        store.save(r1)
        store.save(r2)

        latest = store.get_latest("owner/repo#42")
        assert latest is not None
        assert latest.record_id == "r2"


# -------------------------------------------------------------------
# Tests — get_history
# -------------------------------------------------------------------

class TestRecordStoreGetHistory:
    """Tests for RecordStore.get_history."""

    def test_empty_history(self, tmp_path):
        store = RecordStore(storage_dir=str(tmp_path))
        assert store.get_history("owner/repo#42") == []

    def test_returns_sorted_ascending(self, tmp_path):
        store = RecordStore(storage_dir=str(tmp_path))
        # Save in reverse order to verify sorting
        r2 = _make_record(
            record_id="r2",
            created_at="2024-01-16T14:00:00",
            version_id="v2",
        )
        r1 = _make_record(
            record_id="r1",
            created_at="2024-01-15T10:00:00",
            version_id="v1",
        )
        r3 = _make_record(
            record_id="r3",
            created_at="2024-01-17T08:00:00",
            version_id="v3",
        )
        store.save(r2)
        store.save(r1)
        store.save(r3)

        history = store.get_history("owner/repo#42")
        assert len(history) == 3
        ids = [r.record_id for r in history]
        assert ids == ["r1", "r2", "r3"]

    def test_latest_matches_last_history(self, tmp_path):
        store = RecordStore(storage_dir=str(tmp_path))
        r1 = _make_record(
            record_id="r1",
            created_at="2024-01-15T10:00:00",
            version_id="v1",
        )
        r2 = _make_record(
            record_id="r2",
            created_at="2024-01-16T14:00:00",
            version_id="v2",
        )
        store.save(r1)
        store.save(r2)

        history = store.get_history("owner/repo#42")
        latest = store.get_latest("owner/repo#42")
        assert latest is not None
        assert latest.record_id == history[-1].record_id

    def test_different_pr_ids_isolated(self, tmp_path):
        store = RecordStore(storage_dir=str(tmp_path))
        r1 = _make_record(
            record_id="r1",
            pr_id="owner/repo#42",
            version_id="v1",
        )
        r2 = _make_record(
            record_id="r2",
            pr_id="owner/repo#99",
            version_id="v2",
        )
        store.save(r1)
        store.save(r2)

        h42 = store.get_history("owner/repo#42")
        h99 = store.get_history("owner/repo#99")
        assert len(h42) == 1
        assert len(h99) == 1
        assert h42[0].record_id == "r1"
        assert h99[0].record_id == "r2"

    def test_skips_invalid_json_files(self, tmp_path):
        store = RecordStore(storage_dir=str(tmp_path))
        record = _make_record()
        store.save(record)

        # Write a corrupt JSON file in the same PR directory
        pr_dir = tmp_path / "github" / "owner" / "repo" / "prs" / "42"
        corrupt = pr_dir / "corrupt.json"
        corrupt.write_text("not valid json", encoding="utf-8")

        history = store.get_history("owner/repo#42")
        assert len(history) == 1
        assert history[0].record_id == "rec-1"
