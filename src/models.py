"""Core data models for the PR Review system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PRInfo:
    """PR information fetched from a platform."""

    platform: str  # "github" | "gitlab" | "codeup"
    pr_id: str
    pr_url: str
    title: str
    description: str
    diff: str  # unified diff format
    source_branch: str
    target_branch: str
    author: str
    version_id: str  # commit SHA or MR version


@dataclass
class ReviewIssue:
    """A single issue found during code review."""

    file_path: str
    line_number: int | None
    severity: str  # "critical" | "warning" | "suggestion"
    category: str  # "quality" | "bug" | "security" | "improvement"
    description: str
    suggestion: str | None


@dataclass
class ReviewResult:
    """Result of an AI code review."""

    summary: str
    issues: list[ReviewIssue]
    reviewed_at: str  # ISO 8601 timestamp


@dataclass
class ReviewDiffReport:
    """Comparison report between two review results."""

    improved: list[dict]
    unresolved: list[dict]
    new_issues: list[dict]


@dataclass
class ReviewRecord:
    """A persisted review record."""

    record_id: str  # UUID
    pr_id: str
    pr_url: str
    platform: str
    version_id: str
    review_result: ReviewResult
    diff_report: ReviewDiffReport | None
    created_at: str  # ISO 8601
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    token_usage_by_group: list[TokenUsageByGroup] | None = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON storage."""
        issues = [
            {
                "file_path": issue.file_path,
                "line_number": issue.line_number,
                "severity": issue.severity,
                "category": issue.category,
                "description": issue.description,
                "suggestion": issue.suggestion,
            }
            for issue in self.review_result.issues
        ]
        result_dict = {
            "summary": self.review_result.summary,
            "issues": issues,
            "reviewed_at": self.review_result.reviewed_at,
        }
        diff_report_dict = None
        if self.diff_report is not None:
            diff_report_dict = {
                "improved": self.diff_report.improved,
                "unresolved": self.diff_report.unresolved,
                "new_issues": self.diff_report.new_issues,
            }
        token_by_group = None
        if self.token_usage_by_group:
            token_by_group = [
                {
                    "group_name": g.group_name,
                    "prompt_tokens": g.prompt_tokens,
                    "completion_tokens": g.completion_tokens,
                    "total_tokens": g.total_tokens,
                }
                for g in self.token_usage_by_group
            ]
        return {
            "record_id": self.record_id,
            "pr_id": self.pr_id,
            "pr_url": self.pr_url,
            "platform": self.platform,
            "version_id": self.version_id,
            "review_result": result_dict,
            "diff_report": diff_report_dict,
            "token_usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "by_group": token_by_group,
            },
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReviewRecord:
        """Deserialize from a plain dict (e.g. loaded from JSON)."""
        result_data = data["review_result"]
        issues = [
            ReviewIssue(
                file_path=i["file_path"],
                line_number=i["line_number"],
                severity=i["severity"],
                category=i["category"],
                description=i["description"],
                suggestion=i["suggestion"],
            )
            for i in result_data["issues"]
        ]
        review_result = ReviewResult(
            summary=result_data["summary"],
            issues=issues,
            reviewed_at=result_data["reviewed_at"],
        )
        diff_report = None
        if data.get("diff_report") is not None:
            dr = data["diff_report"]
            diff_report = ReviewDiffReport(
                improved=dr["improved"],
                unresolved=dr["unresolved"],
                new_issues=dr["new_issues"],
            )
        tu = data.get("token_usage") or {}
        by_group_raw = tu.get("by_group")
        by_group = None
        if by_group_raw:
            by_group = [
                TokenUsageByGroup(
                    group_name=g["group_name"],
                    prompt_tokens=g["prompt_tokens"],
                    completion_tokens=g["completion_tokens"],
                    total_tokens=g["total_tokens"],
                )
                for g in by_group_raw
            ]
        return cls(
            record_id=data["record_id"],
            pr_id=data["pr_id"],
            pr_url=data["pr_url"],
            platform=data["platform"],
            version_id=data["version_id"],
            review_result=review_result,
            diff_report=diff_report,
            created_at=data["created_at"],
            prompt_tokens=tu.get("prompt_tokens", 0),
            completion_tokens=tu.get("completion_tokens", 0),
            total_tokens=tu.get("total_tokens", 0),
            token_usage_by_group=by_group,
        )


@dataclass
class ReviewOptions:
    """Options controlling the review process."""

    template_path: str | None = None
    write_back: bool = True
    exclude_patterns: list[str] | None = None
    use_default_excludes: bool = True


@dataclass
class FileGroup:
    """文件关联域分组。"""

    name: str
    file_paths: list[str]
    file_diffs: list[str]
    total_chars: int


@dataclass
class Batch:
    """组内二次分片批次。"""

    group_name: str
    batch_index: int
    file_paths: list[str]
    diff_content: str
    char_count: int


@dataclass
class SymbolEntry:
    """符号索引条目。"""

    name: str
    signature: str
    file_path: str
    line_number: int
    kind: str  # "function" | "method" | "interface" | "struct" | "class" | "service" | "rpc" | "message"
    language: str  # "go" | "proto" | "typescript" | "python"


@dataclass
class SubAgentResult:
    """子 Agent 审查结果。"""

    group_name: str
    batch_index: int
    result: ReviewResult | None
    error: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class TokenUsageByGroup:
    """分组维度 token 消耗。"""

    group_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ReviewOutput:
    """Output of a complete review run."""

    review_result: ReviewResult
    diff_report: ReviewDiffReport | None
    formatted_comment: str
    written_back: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    token_usage_by_group: list[TokenUsageByGroup] | None = None


@dataclass
class FilterResult:
    """Result of filtering a diff by exclude patterns."""

    filtered_diff: str
    excluded_files: list[dict]  # [{"file_path": str, "matched_pattern": str}]
    included_file_count: int
    excluded_file_count: int
