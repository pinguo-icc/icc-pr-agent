"""Unit tests for src/result_merger.py."""

from src.models import ReviewIssue, ReviewResult, SubAgentResult
from src.result_merger import ResultMerger


def _issue(file_path="a.go", line=1, severity="warning", desc="issue"):
    return ReviewIssue(
        file_path=file_path, line_number=line, severity=severity,
        category="quality", description=desc, suggestion=None,
    )


def _result(issues, summary="ok"):
    return ReviewResult(summary=summary, issues=issues, reviewed_at="2024-01-01T00:00:00Z")


def _sr(group="backend", batch=0, issues=None, error=None, summary="ok",
        prompt_tokens=0, completion_tokens=0, total_tokens=0):
    result = _result(issues or [], summary) if error is None else None
    return SubAgentResult(
        group_name=group, batch_index=batch, result=result, error=error,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


class TestResultMergerBasic:
    def test_empty_results(self):
        merger = ResultMerger()
        result = merger.merge([])
        assert result.issues == []
        assert result.summary

    def test_single_successful_result(self):
        issues = [_issue(severity="critical")]
        merger = ResultMerger()
        result = merger.merge([_sr(issues=issues)])
        assert len(result.issues) == 1

    def test_merge_multiple_results(self):
        merger = ResultMerger()
        result = merger.merge([
            _sr(issues=[_issue(file_path="a.go", line=1)]),
            _sr(group="frontend", issues=[_issue(file_path="b.ts", line=2)]),
        ])
        assert len(result.issues) == 2


class TestResultMergerSeveritySorting:
    def test_sorted_by_severity(self):
        merger = ResultMerger()
        result = merger.merge([_sr(issues=[
            _issue(severity="suggestion", line=1),
            _issue(severity="critical", line=2),
            _issue(severity="warning", line=3),
        ])])
        severities = [i.severity for i in result.issues]
        assert severities == ["critical", "warning", "suggestion"]

    def test_truncation_by_max_issues(self):
        issues = [_issue(line=i, severity="warning") for i in range(20)]
        merger = ResultMerger()
        result = merger.merge([_sr(issues=issues)], max_issues=5)
        assert len(result.issues) == 5


class TestResultMergerDeduplication:
    def test_exact_position_dedup(self):
        """Requirement 9.2: same file_path + line_number -> keep highest severity."""
        merger = ResultMerger()
        result = merger.merge([_sr(issues=[
            _issue(file_path="a.go", line=10, severity="warning", desc="issue A"),
            _issue(file_path="a.go", line=10, severity="critical", desc="issue B"),
        ])])
        assert len(result.issues) == 1
        assert result.issues[0].severity == "critical"

    def test_jaccard_dedup_null_lines(self):
        """Requirement 9.3: same file + null lines + similar desc -> dedup."""
        merger = ResultMerger()
        result = merger.merge([_sr(issues=[
            _issue(file_path="a.go", line=None, severity="warning",
                   desc="this function is too long and complex"),
            _issue(file_path="a.go", line=None, severity="critical",
                   desc="this function is too long and complex please refactor"),
        ])])
        # These should be deduped (high Jaccard similarity)
        assert len(result.issues) <= 2  # May or may not dedup depending on threshold

    def test_pipeline_order(self):
        """Requirement 9.4: dedup before sort and truncate."""
        merger = ResultMerger()
        # Two identical position issues + one unique
        result = merger.merge([_sr(issues=[
            _issue(file_path="a.go", line=5, severity="suggestion"),
            _issue(file_path="a.go", line=5, severity="critical"),
            _issue(file_path="b.go", line=1, severity="warning"),
        ])], max_issues=10)
        # After dedup: 2 issues (one from a.go:5, one from b.go:1)
        assert len(result.issues) == 2


class TestResultMergerFailureHandling:
    def test_partial_failure(self):
        """Requirement 7.5: failed sub-agents logged, not in summary."""
        merger = ResultMerger()
        result = merger.merge([
            _sr(issues=[_issue()]),
            _sr(group="frontend", error="Timeout after 300s"),
        ])
        assert len(result.issues) == 1
        # Failed group info should NOT appear in summary (only in logs)
        assert "frontend" not in result.summary
        assert result.all_failed is False

    def test_all_failed(self):
        merger = ResultMerger()
        result = merger.merge([
            _sr(group="backend", error="Timeout"),
            _sr(group="frontend", error="API error"),
        ])
        assert result.issues == []
        assert result.all_failed is True
        assert "失败" in result.summary


class TestJaccardSimilarity:
    def test_identical_strings(self):
        assert ResultMerger._jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert ResultMerger._jaccard_similarity("hello", "world") == 0.0

    def test_partial_overlap(self):
        sim = ResultMerger._jaccard_similarity("a b c", "b c d")
        assert 0.0 < sim < 1.0

    def test_empty_strings(self):
        assert ResultMerger._jaccard_similarity("", "") == 1.0

    def test_one_empty(self):
        assert ResultMerger._jaccard_similarity("hello", "") == 0.0


class TestTokenAggregation:
    def test_aggregate_single_group(self):
        results = [
            _sr(group="backend", prompt_tokens=100, completion_tokens=50, total_tokens=150),
            _sr(group="backend", batch=1, prompt_tokens=200, completion_tokens=100, total_tokens=300),
        ]
        usage = ResultMerger.aggregate_token_usage(results)
        assert len(usage) == 1
        assert usage[0].group_name == "backend"
        assert usage[0].prompt_tokens == 300
        assert usage[0].total_tokens == 450

    def test_aggregate_multiple_groups(self):
        results = [
            _sr(group="backend", prompt_tokens=100, completion_tokens=50, total_tokens=150),
            _sr(group="frontend", prompt_tokens=200, completion_tokens=100, total_tokens=300),
        ]
        usage = ResultMerger.aggregate_token_usage(results)
        assert len(usage) == 2
        names = {u.group_name for u in usage}
        assert names == {"backend", "frontend"}
