"""Result merger for combining sub-agent review results."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.models import ReviewIssue, ReviewResult, SubAgentResult, TokenUsageByGroup

logger = logging.getLogger(__name__)


class ResultMerger:
    """审查结果合并器。"""

    SEVERITY_ORDER: dict[str, int] = {"critical": 0, "warning": 1, "suggestion": 2}
    JACCARD_THRESHOLD: float = 0.8

    def merge(
        self,
        sub_results: list[SubAgentResult],
        max_issues: int = 10,
    ) -> ReviewResult:
        """合并子 Agent 结果。

        处理顺序：合并 → 去重 → 排序 → 截断 → 合并 summary。
        """
        all_issues: list[ReviewIssue] = []
        summaries: list[str] = []
        failed_groups: list[str] = []

        for sr in sub_results:
            if sr.result is not None:
                all_issues.extend(sr.result.issues)
                if sr.result.summary:
                    summaries.append(sr.result.summary)
            elif sr.error:
                failed_groups.append(f"{sr.group_name}(batch {sr.batch_index})")
                logger.error(
                    "Sub-agent failed: group=%s batch=%d error=%s",
                    sr.group_name, sr.batch_index, sr.error,
                )
            else:
                # result is None but error is empty/falsy (e.g. TimeoutError)
                failed_groups.append(f"{sr.group_name}(batch {sr.batch_index})")
                logger.error(
                    "Sub-agent failed with no result: group=%s batch=%d",
                    sr.group_name, sr.batch_index,
                )

        # Pipeline: deduplicate → sort → truncate
        deduped = self._deduplicate(all_issues)
        sorted_issues = self._sort_by_severity(deduped)
        truncated = sorted_issues[:max_issues]

        # Build summary
        summary_parts = summaries.copy()
        if failed_groups:
            summary_parts.append(
                f"以下分组审查失败: {', '.join(failed_groups)}"
            )
        summary = "\n\n".join(summary_parts) if summary_parts else "审查完成，未发现问题。"

        return ReviewResult(
            summary=summary,
            issues=truncated,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _deduplicate(issues: list[ReviewIssue]) -> list[ReviewIssue]:
        """去重逻辑：
        - file_path + line_number 完全相同 → 保留 severity 最高的
        - file_path 相同 + line_number 均为 null + Jaccard(description) ≥ 0.8 → 保留 severity 最高的
        """
        if not issues:
            return []

        severity_rank = ResultMerger.SEVERITY_ORDER
        result: list[ReviewIssue] = []

        # Group by file_path for efficient comparison
        by_file: dict[str, list[ReviewIssue]] = {}
        for issue in issues:
            by_file.setdefault(issue.file_path, []).append(issue)

        for file_path, file_issues in by_file.items():
            # Separate issues with and without line numbers
            with_line: dict[int, ReviewIssue] = {}
            without_line: list[ReviewIssue] = []

            for issue in file_issues:
                if issue.line_number is not None:
                    existing = with_line.get(issue.line_number)
                    if existing is None:
                        with_line[issue.line_number] = issue
                    else:
                        # Keep higher severity (lower rank number)
                        if severity_rank.get(issue.severity, 99) < severity_rank.get(existing.severity, 99):
                            with_line[issue.line_number] = issue
                else:
                    without_line.append(issue)

            result.extend(with_line.values())

            # Deduplicate null-line issues by Jaccard similarity
            kept: list[ReviewIssue] = []
            for issue in without_line:
                is_dup = False
                for i, existing in enumerate(kept):
                    sim = ResultMerger._jaccard_similarity(
                        issue.description, existing.description,
                    )
                    if sim >= ResultMerger.JACCARD_THRESHOLD:
                        is_dup = True
                        if severity_rank.get(issue.severity, 99) < severity_rank.get(existing.severity, 99):
                            kept[i] = issue
                        break
                if not is_dup:
                    kept.append(issue)
            result.extend(kept)

        return result

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """计算两个字符串的 Jaccard 相似度（基于词集合）。"""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _sort_by_severity(issues: list[ReviewIssue]) -> list[ReviewIssue]:
        """按 severity 优先级排序：critical > warning > suggestion。"""
        return sorted(
            issues,
            key=lambda i: ResultMerger.SEVERITY_ORDER.get(i.severity, 99),
        )

    @staticmethod
    def aggregate_token_usage(
        sub_results: list[SubAgentResult],
    ) -> list[TokenUsageByGroup]:
        """按 group_name 聚合 token 消耗。"""
        by_group: dict[str, dict[str, int]] = {}
        for sr in sub_results:
            g = by_group.setdefault(sr.group_name, {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            })
            g["prompt_tokens"] += sr.prompt_tokens
            g["completion_tokens"] += sr.completion_tokens
            g["total_tokens"] += sr.total_tokens

        return [
            TokenUsageByGroup(
                group_name=name,
                prompt_tokens=data["prompt_tokens"],
                completion_tokens=data["completion_tokens"],
                total_tokens=data["total_tokens"],
            )
            for name, data in by_group.items()
        ]
