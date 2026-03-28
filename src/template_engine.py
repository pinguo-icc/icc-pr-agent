"""Template engine for rendering review results using Jinja2."""

from __future__ import annotations

import os
import re

from jinja2 import Environment, FileSystemLoader

from src.exceptions import TemplateNotFoundError
from src.models import PRInfo, ReviewDiffReport, ReviewResult, TokenUsageByGroup


def _linebreak_sentences(text: str) -> str:
    """Split a dense summary into Markdown-friendly lines.

    Handles numbered lists like ``1) ... 2) ...`` and Chinese
    semicolons / periods so each point renders on its own line.
    """
    # Turn "1) " / "2) " style inline lists into Markdown line items
    text = re.sub(r"\s*(\d+)\)\s*", r"\n\n\1. ", text)
    # Also handle "；" as a sentence break
    text = text.replace("；", "；\n\n")
    return text.strip()


class TemplateEngine:
    """Renders review results into formatted text using Jinja2 templates."""

    DEFAULT_TEMPLATE_PATH = os.path.join("templates", "default.md.j2")

    def render(
        self,
        review_result: ReviewResult,
        pr_info: PRInfo,
        diff_report: ReviewDiffReport | None = None,
        template_path: str | None = None,
        excluded_files: list[dict] | None = None,
        token_usage: dict | None = None,
        token_usage_by_group: list[TokenUsageByGroup] | None = None,
        tools_used: list[str] | None = None,
    ) -> str:
        """Render review result using a Jinja2 template.

        Args:
            review_result: The AI review result.
            pr_info: PR metadata.
            diff_report: Optional diff comparison report.
            template_path: Path to a custom template. Uses default if None.
            excluded_files: Optional list of excluded file dicts.

        Returns:
            Rendered string.

        Raises:
            TemplateNotFoundError: If the specified template file does not exist.
        """
        path = template_path if template_path is not None else self.DEFAULT_TEMPLATE_PATH

        if not os.path.isfile(path):
            raise TemplateNotFoundError(f"Template not found: {path}")

        template_dir = os.path.dirname(os.path.abspath(path))
        template_name = os.path.basename(path)

        env = Environment(
            loader=FileSystemLoader(template_dir),
            keep_trailing_newline=True,
        )
        env.filters["linebreak_sentences"] = _linebreak_sentences
        template = env.get_template(template_name)

        return template.render(
            pr_info=pr_info,
            result=review_result,
            diff_report=diff_report,
            excluded_files=excluded_files,
            token_usage=token_usage,
            token_usage_by_group=token_usage_by_group,
            tools_used=tools_used,
        )
