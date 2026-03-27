"""Property-based tests for ResultMerger."""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.models import ReviewIssue, ReviewResult, SubAgentResult
from src.result_merger import ResultMerger

_SEVERITIES = ["critical", "warning", "suggestion"]
_SEVERITY_RANK = {"critical": 0, "warning": 1, "suggestion": 2}


def _issue_strategy(unique_lines=True):
    return st.builds(
        ReviewIssue,
        file_path=st.from_regex(r"[a-z]{1,5}/[a-z]{1,8}\.(go|py|ts)", fullmatch=True),
        line_number=st.integers(min_value=1, max_value=10000) if unique_lines else st.just(None),
        severity=st.sampled_from(_SEVERITIES),
        category=st.sampled_from(["quality", "bug", "security", "improvement"]),
        description=st.from_regex(r"[a-z ]{5,30}", fullmatch=True),
        suggestion=st.none(),
    )


def _sub_result_strategy(success=True):
    if success:
        issues = st.lists(_issue_strategy(), min_size=0, max_size=5)
        return issues.map(lambda iss: SubAgentResult(
            group_name="test", batch_index=0,
            result=ReviewResult(summary="ok", issues=iss, reviewed_at="2024-01-01T00:00:00Z"),
            error=None,
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
        ))
    return st.just(SubAgentResult(
        group_name="failed", batch_index=0, result=None, error="timeout",
    ))


# --- Property 8: Severity 优先级截断 ---
# Feature: sub-agent-review, Property 8: Severity 优先级截断

@given(
    issues=st.lists(_issue_strategy(), min_size=0, max_size=30),
    max_issues=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=100)
def test_property_8_severity_truncation(issues, max_issues):
    """Truncated list <= max_issues, sorted by severity.

    Validates: Requirements 6.3
    """
    sorted_issues = ResultMerger._sort_by_severity(issues)
    truncated = sorted_issues[:max_issues]

    assert len(truncated) <= max_issues

    # Verify severity ordering
    for i in range(len(truncated) - 1):
        rank_a = _SEVERITY_RANK.get(truncated[i].severity, 99)
        rank_b = _SEVERITY_RANK.get(truncated[i + 1].severity, 99)
        assert rank_a <= rank_b


# --- Property 9: 合并产生有序的成功结果并集 ---
# Feature: sub-agent-review, Property 9: 合并产生有序的成功结果并集


def _unique_issue_strategy():
    """Issues with unique file_path+line combos to avoid dedup interference."""
    counter = {"n": 0}

    def make():
        counter["n"] += 1
        return ReviewIssue(
            file_path=f"file_{counter['n']}.go",
            line_number=counter["n"],
            severity="warning",
            category="quality",
            description=f"issue {counter['n']}",
            suggestion=None,
        )
    return make


@given(
    num_results=st.integers(min_value=1, max_value=5),
    issues_per=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=100)
def test_property_9_merge_union(num_results, issues_per):
    """Merged issues contain union of all successful sub-results.

    Validates: Requirements 7.1, 7.2
    """
    factory = _unique_issue_strategy()
    sub_results = []
    expected_count = 0
    for i in range(num_results):
        issues = [factory() for _ in range(issues_per)]
        expected_count += len(issues)
        sub_results.append(SubAgentResult(
            group_name=f"g{i}", batch_index=0,
            result=ReviewResult(summary="ok", issues=issues, reviewed_at="2024-01-01T00:00:00Z"),
            error=None,
        ))

    merger = ResultMerger()
    result = merger.merge(sub_results, max_issues=1000)
    assert len(result.issues) == expected_count


# --- Property 10: 合并容忍子 Agent 失败 ---
# Feature: sub-agent-review, Property 10: 合并容忍子 Agent 失败

@given(
    num_success=st.integers(min_value=0, max_value=3),
    num_failed=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=100)
def test_property_10_tolerates_failures(num_success, num_failed):
    """Merge completes with failed sub-agents; summary mentions failed groups.

    Validates: Requirements 7.5
    """
    factory = _unique_issue_strategy()
    sub_results = []
    for i in range(num_success):
        sub_results.append(SubAgentResult(
            group_name=f"ok_{i}", batch_index=0,
            result=ReviewResult(summary="fine", issues=[factory()], reviewed_at="2024-01-01T00:00:00Z"),
            error=None,
        ))
    for i in range(num_failed):
        sub_results.append(SubAgentResult(
            group_name=f"fail_{i}", batch_index=0,
            result=None, error="timeout",
        ))

    merger = ResultMerger()
    result = merger.merge(sub_results, max_issues=1000)

    # Should not raise
    assert isinstance(result, ReviewResult)
    assert result.summary  # non-empty

    # Failed groups mentioned in summary
    for i in range(num_failed):
        assert f"fail_{i}" in result.summary


# --- Property 11: 合并结果结构兼容 ---
# Feature: sub-agent-review, Property 11: 合并结果结构兼容

@given(sub_results=st.lists(_sub_result_strategy(), min_size=0, max_size=5))
@settings(max_examples=100)
def test_property_11_result_structure(sub_results):
    """Merged result has valid structure.

    Validates: Requirements 7.4, 13.3
    """
    merger = ResultMerger()
    result = merger.merge(sub_results, max_issues=100)

    assert isinstance(result.summary, str)
    assert len(result.summary) > 0
    assert isinstance(result.issues, list)

    for issue in result.issues:
        assert issue.file_path
        assert issue.severity in _SEVERITIES
        assert issue.category
        assert issue.description


# --- Property 15: 精确位置去重保留最高 Severity ---
# Feature: sub-agent-review, Property 15: 精确位置去重保留最高 Severity

@given(
    file_path=st.from_regex(r"[a-z]{1,5}\.(go|py)", fullmatch=True),
    line=st.integers(min_value=1, max_value=1000),
    sev_a=st.sampled_from(_SEVERITIES),
    sev_b=st.sampled_from(_SEVERITIES),
)
@settings(max_examples=100)
def test_property_15_exact_position_dedup(file_path, line, sev_a, sev_b):
    """Same file_path + line_number -> keep highest severity only.

    Validates: Requirements 9.2
    """
    issues = [
        ReviewIssue(file_path=file_path, line_number=line, severity=sev_a,
                    category="quality", description="issue A", suggestion=None),
        ReviewIssue(file_path=file_path, line_number=line, severity=sev_b,
                    category="quality", description="issue B", suggestion=None),
    ]
    deduped = ResultMerger._deduplicate(issues)

    # Should have exactly 1 issue for this position
    matching = [i for i in deduped if i.file_path == file_path and i.line_number == line]
    assert len(matching) == 1

    # Kept issue should have the highest severity
    best_sev = min(sev_a, sev_b, key=lambda s: _SEVERITY_RANK[s])
    assert matching[0].severity == best_sev


# --- Property 16: Jaccard 相似度去重 ---
# Feature: sub-agent-review, Property 16: Jaccard 相似度去重

@given(
    base_words=st.lists(st.from_regex(r"[a-z]{3,8}", fullmatch=True), min_size=5, max_size=10),
    sev_a=st.sampled_from(_SEVERITIES),
    sev_b=st.sampled_from(_SEVERITIES),
)
@settings(max_examples=100)
def test_property_16_jaccard_dedup(base_words, sev_a, sev_b):
    """Similar descriptions (Jaccard >= 0.8) with null lines -> dedup.

    Validates: Requirements 9.3
    """
    desc_a = " ".join(base_words)
    desc_b = " ".join(base_words)  # Identical = Jaccard 1.0

    issues = [
        ReviewIssue(file_path="a.go", line_number=None, severity=sev_a,
                    category="quality", description=desc_a, suggestion=None),
        ReviewIssue(file_path="a.go", line_number=None, severity=sev_b,
                    category="quality", description=desc_b, suggestion=None),
    ]
    deduped = ResultMerger._deduplicate(issues)

    null_line = [i for i in deduped if i.file_path == "a.go" and i.line_number is None]
    assert len(null_line) == 1

    best_sev = min(sev_a, sev_b, key=lambda s: _SEVERITY_RANK[s])
    assert null_line[0].severity == best_sev


# --- Property 17: Token 消耗分组聚合 ---
# Feature: sub-agent-review, Property 17: Token 消耗分组聚合

@given(
    data=st.lists(
        st.tuples(
            st.from_regex(r"[a-z]{2,6}", fullmatch=True),  # group_name
            st.integers(min_value=0, max_value=10000),  # prompt
            st.integers(min_value=0, max_value=10000),  # completion
        ),
        min_size=1, max_size=10,
    )
)
@settings(max_examples=100)
def test_property_17_token_aggregation(data):
    """Token usage aggregated by group equals sum of sub-results.

    Validates: Requirements 11.1, 11.2
    """
    sub_results = []
    for i, (group, prompt, completion) in enumerate(data):
        sub_results.append(SubAgentResult(
            group_name=group, batch_index=i, result=None, error=None,
            prompt_tokens=prompt, completion_tokens=completion,
            total_tokens=prompt + completion,
        ))

    usage = ResultMerger.aggregate_token_usage(sub_results)
    usage_map = {u.group_name: u for u in usage}

    # Verify each group's totals
    from collections import defaultdict
    expected: dict[str, dict[str, int]] = defaultdict(lambda: {"p": 0, "c": 0, "t": 0})
    for sr in sub_results:
        expected[sr.group_name]["p"] += sr.prompt_tokens
        expected[sr.group_name]["c"] += sr.completion_tokens
        expected[sr.group_name]["t"] += sr.total_tokens

    for gname, exp in expected.items():
        assert gname in usage_map
        assert usage_map[gname].prompt_tokens == exp["p"]
        assert usage_map[gname].completion_tokens == exp["c"]
        assert usage_map[gname].total_tokens == exp["t"]


# --- Property 18: 总 Token 等于分组 Token 之和 ---
# Feature: sub-agent-review, Property 18: 总 Token 等于分组 Token 之和

@given(
    data=st.lists(
        st.tuples(
            st.from_regex(r"[a-z]{2,6}", fullmatch=True),
            st.integers(min_value=0, max_value=10000),
            st.integers(min_value=0, max_value=10000),
        ),
        min_size=1, max_size=10,
    )
)
@settings(max_examples=100)
def test_property_18_total_equals_sum(data):
    """Total tokens equals sum of all group tokens.

    Validates: Requirements 11.4
    """
    sub_results = []
    for i, (group, prompt, completion) in enumerate(data):
        sub_results.append(SubAgentResult(
            group_name=group, batch_index=i, result=None, error=None,
            prompt_tokens=prompt, completion_tokens=completion,
            total_tokens=prompt + completion,
        ))

    usage = ResultMerger.aggregate_token_usage(sub_results)

    total_prompt = sum(u.prompt_tokens for u in usage)
    total_completion = sum(u.completion_tokens for u in usage)
    total_tokens = sum(u.total_tokens for u in usage)

    expected_prompt = sum(sr.prompt_tokens for sr in sub_results)
    expected_completion = sum(sr.completion_tokens for sr in sub_results)
    expected_total = sum(sr.total_tokens for sr in sub_results)

    assert total_prompt == expected_prompt
    assert total_completion == expected_completion
    assert total_tokens == expected_total
