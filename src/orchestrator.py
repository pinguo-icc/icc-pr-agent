"""ReviewOrchestrator — coordinates the full PR review workflow."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.ai_reviewer import AIReviewer
from src.config import Config
from src.diff_comparator import DiffComparator
from src.exceptions import (
    AllFilesExcludedError,
    CommentWriteBackError,
    EmptyDiffError,
)
from src.file_filter import FileFilter
from src.logger import get_logger
from src.models import (
    ReviewOptions,
    ReviewOutput,
    ReviewRecord,
)
from src.platform.factory import PlatformFactory
from src.record_store import RecordStore
from src.template_engine import TemplateEngine

logger = get_logger(__name__)


class ReviewOrchestrator:
    """Orchestrate the complete PR review pipeline."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._ai_reviewer = AIReviewer(config)
        self._template_engine = TemplateEngine()
        self._diff_comparator = DiffComparator()
        self._record_store = RecordStore(config.review_storage_dir)
        self._file_filter = FileFilter()

    def run(
        self, pr_url: str, options: ReviewOptions
    ) -> ReviewOutput:
        """Execute the full review flow for a PR.

        Raises:
            EmptyDiffError: The PR has no code changes.
            AllFilesExcludedError: Every file was excluded.
        """
        start_time = datetime.now(timezone.utc).isoformat()

        # a. Log review start
        adapter = PlatformFactory.create_adapter(pr_url)
        platform = PlatformFactory.detect_platform(pr_url)
        logger.info(
            "Review started at %s | platform=%s | url=%s",
            start_time,
            platform,
            pr_url,
        )

        # c. Fetch PR info
        pr_info = adapter.fetch_pr_info(pr_url)

        # d. Build FileFilter with merged patterns
        config_patterns = FileFilter.load_patterns_from_config()
        env_patterns = FileFilter.load_patterns_from_env()
        user_patterns = list(options.exclude_patterns or [])
        merged_patterns = list(
            dict.fromkeys(
                user_patterns + config_patterns + env_patterns
            )
        )
        file_filter = FileFilter(
            exclude_patterns=merged_patterns if merged_patterns else None,
            use_defaults=options.use_default_excludes,
        )

        # e. Filter diff
        filter_result = file_filter.filter_diff(pr_info.diff)

        # f/g. Check empty results
        if (
            not filter_result.filtered_diff.strip()
            and filter_result.excluded_file_count > 0
        ):
            logger.warning(
                "All files excluded for PR %s", pr_info.pr_id
            )
            raise AllFilesExcludedError(
                "所有变更文件均被排除，无需审查"
            )

        if not filter_result.filtered_diff.strip():
            logger.warning(
                "Empty diff for PR %s", pr_info.pr_id
            )
            raise EmptyDiffError("无代码变更，无需审查")

        # h. Update diff with filtered version
        pr_info.diff = filter_result.filtered_diff

        # i. Retrieve previous record
        previous_record = self._record_store.get_latest(
            pr_info.pr_id
        )

        # j. AI review
        review_result = self._ai_reviewer.review(pr_info)

        # k. Compare with previous if exists
        diff_report = None
        if previous_record is not None:
            diff_report = self._diff_comparator.compare(
                previous_record.review_result, review_result
            )

        # l. Render template
        token_usage = {
            "prompt_tokens": self._ai_reviewer.total_prompt_tokens,
            "completion_tokens": self._ai_reviewer.total_completion_tokens,
            "total_tokens": self._ai_reviewer.total_tokens,
        }
        formatted_comment = self._template_engine.render(
            review_result,
            pr_info,
            diff_report,
            options.template_path,
            filter_result.excluded_files,
            token_usage=token_usage,
            token_usage_by_group=self._ai_reviewer.token_usage_by_group or None,
            tools_used=self._ai_reviewer.tools_used or None,
            skills_loaded=self._ai_reviewer.skills_loaded or None,
            model_name=self._ai_reviewer.actual_model,
        )

        # m. Save record
        record = ReviewRecord(
            record_id=str(uuid.uuid4()),
            pr_id=pr_info.pr_id,
            pr_url=pr_info.pr_url,
            platform=pr_info.platform,
            version_id=pr_info.version_id,
            review_result=review_result,
            diff_report=diff_report,
            created_at=datetime.now(timezone.utc).isoformat(),
            prompt_tokens=self._ai_reviewer.total_prompt_tokens,
            completion_tokens=self._ai_reviewer.total_completion_tokens,
            total_tokens=self._ai_reviewer.total_tokens,
            token_usage_by_group=self._ai_reviewer.token_usage_by_group or None,
            trace=self._ai_reviewer.traces or None,
        )
        self._record_store.save(record)

        # n. Write back comment (skip if all sub-agents failed)
        written_back = False
        if review_result.all_failed:
            logger.error(
                "All review sub-agents failed for %s, "
                "skipping PR comment write-back. summary=%s",
                pr_info.pr_id,
                review_result.summary,
            )
        elif options.write_back:
            try:
                adapter.post_comment(pr_url, formatted_comment)
                written_back = True
            except CommentWriteBackError as exc:
                logger.error(
                    "Failed to write back comment: %s", exc
                )

        # o. Log review end
        end_time = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Review ended at %s | platform=%s | pr_id=%s",
            end_time,
            platform,
            pr_info.pr_id,
        )

        # p. Return output
        return ReviewOutput(
            review_result=review_result,
            diff_report=diff_report,
            formatted_comment=formatted_comment,
            written_back=written_back,
            prompt_tokens=self._ai_reviewer.total_prompt_tokens,
            completion_tokens=self._ai_reviewer.total_completion_tokens,
            total_tokens=self._ai_reviewer.total_tokens,
            token_usage_by_group=self._ai_reviewer.token_usage_by_group or None,
        )
