"""CLIRunner — command-line entry point for PR review."""

from __future__ import annotations

import argparse
import sys

from src.config import Config
from src.exceptions import (
    AllFilesExcludedError,
    EmptyDiffError,
    PRReviewError,
)
from src.models import ReviewOptions
from src.orchestrator import ReviewOrchestrator


class CLIRunner:
    """Parse CLI arguments and drive the review pipeline."""

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="pr.py",
            description="AI-powered PR code review tool",
        )
        parser.add_argument(
            "pr_url",
            nargs="?",
            default=None,
            help="PR link to review",
        )
        parser.add_argument(
            "--template",
            default=None,
            help="Custom template file path",
        )
        parser.add_argument(
            "--no-write-back",
            action="store_true",
            default=False,
            help="Disable writing review back to platform",
        )
        parser.add_argument(
            "--exclude",
            action="append",
            default=None,
            help=(
                "Exclude pattern (repeatable, "
                'e.g. --exclude "*.lock" --exclude "*.png")'
            ),
        )
        parser.add_argument(
            "--no-default-exclude",
            action="store_true",
            default=False,
            help="Disable default exclude patterns",
        )
        return parser

    def run(self, args: list[str] | None = None) -> int:
        """Parse *args* and execute the review.

        Returns 0 on success, 1 on failure.
        """
        parser = self._build_parser()
        parsed = parser.parse_args(args)

        if parsed.pr_url is None:
            parser.print_usage(sys.stderr)
            return 1

        options = ReviewOptions(
            template_path=parsed.template,
            write_back=not parsed.no_write_back,
            exclude_patterns=parsed.exclude,
            use_default_excludes=not parsed.no_default_exclude,
        )

        try:
            config = Config.from_env()

            # Initialize Langfuse (same as server.py lifespan)
            from src.langfuse_integration import init_langfuse
            init_langfuse(config)

            orchestrator = ReviewOrchestrator(config)

            print(f"🔍 开始审查 PR: {parsed.pr_url}", flush=True)
            print("📥 获取 PR 信息...", flush=True)

            output = orchestrator.run(
                parsed.pr_url, options
            )

            if output.review_result.all_failed:
                print("❌ 所有分组审查失败，请查看日志了解详情", file=sys.stderr)
                return 1

            print("✅ 审查完成", flush=True)
            print(output.formatted_comment)

            if output.written_back:
                print("📤 审查结果已写回 PR")

        except (EmptyDiffError, AllFilesExcludedError) as exc:
            print(f"⚠️ {exc}", file=sys.stderr)
            return 1
        except PRReviewError as exc:
            print(f"❌ {exc}", file=sys.stderr)
            return 1

        return 0
