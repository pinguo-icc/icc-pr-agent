"""Unit tests for TemplateEngine."""

import os

import pytest

from src.exceptions import TemplateNotFoundError
from src.models import (
    PRInfo,
    ReviewDiffReport,
    ReviewIssue,
    ReviewResult,
    TokenUsageByGroup,
)
from src.template_engine import TemplateEngine


def _pr_info(**overrides) -> PRInfo:
    defaults = dict(
        platform="github",
        pr_id="123",
        pr_url="https://github.com/owner/repo/pull/123",
        title="Fix login bug",
        description="Fixes the login timeout issue",
        diff="diff --git a/app.py b/app.py",
        source_branch="fix/login",
        target_branch="main",
        author="dev-user",
        version_id="abc123",
    )
    defaults.update(overrides)
    return PRInfo(**defaults)


def _review_result(**overrides) -> ReviewResult:
    defaults = dict(
        summary="Found 1 issue",
        issues=[],
        reviewed_at="2024-06-01T12:00:00Z",
    )
    defaults.update(overrides)
    return ReviewResult(**defaults)


class TestTemplateEngine:
    def setup_method(self) -> None:
        self.engine = TemplateEngine()

    # ---- default template rendering ----

    def test_render_with_default_template(self) -> None:
        pr = _pr_info()
        result = _review_result()
        output = self.engine.render(result, pr)

        assert pr.title in output
        assert pr.source_branch in output
        assert pr.target_branch in output
        assert pr.author in output
        assert result.summary in output
        assert result.reviewed_at in output

    def test_render_includes_issues(self) -> None:
        issue = ReviewIssue(
            file_path="src/app.py",
            line_number=42,
            severity="critical",
            category="bug",
            description="Null pointer dereference",
            suggestion="Add null check",
        )
        result = _review_result(issues=[issue])
        output = self.engine.render(result, _pr_info())

        assert "src/app.py" in output
        assert "Null pointer dereference" in output
        assert "Add null check" in output

    def test_render_with_no_issues(self) -> None:
        output = self.engine.render(_review_result(), _pr_info())
        # The default template shows a "no issues" message
        assert "未发现问题" in output or "0" in output

    def test_render_with_diff_report(self) -> None:
        diff_report = ReviewDiffReport(
            improved=[{"description": "fixed import", "resolution": "已修复"}],
            unresolved=[{"description": "missing doc", "file_path": "a.py"}],
            new_issues=[
                {
                    "description": "hardcoded secret",
                    "file_path": "b.py",
                    "severity": "critical",
                }
            ],
        )
        output = self.engine.render(
            _review_result(), _pr_info(), diff_report=diff_report
        )

        # Template should render without error even with diff_report
        assert "AI Code Review" in output

    def test_render_with_excluded_files(self) -> None:
        excluded = [
            {"file_path": "package-lock.json", "matched_pattern": "*.lock"},
        ]
        output = self.engine.render(
            _review_result(), _pr_info(), excluded_files=excluded
        )

        assert "package-lock.json" in output

    # ---- custom template path ----

    def test_render_with_custom_template(self, tmp_path) -> None:
        tpl = tmp_path / "custom.md.j2"
        tpl.write_text("PR: {{ pr_info.title }} | {{ result.summary }}")

        output = self.engine.render(
            _review_result(), _pr_info(), template_path=str(tpl)
        )
        assert "Fix login bug" in output
        assert "Found 1 issue" in output

    def test_custom_template_receives_all_variables(self, tmp_path) -> None:
        tpl = tmp_path / "vars.md.j2"
        tpl.write_text(
            "{% if diff_report %}DIFF{% endif %}"
            "{% if excluded_files %}EXCL{% endif %}"
        )
        diff_report = ReviewDiffReport([], [], [])
        excluded = [{"file_path": "a.lock", "matched_pattern": "*.lock"}]

        output = self.engine.render(
            _review_result(),
            _pr_info(),
            diff_report=diff_report,
            template_path=str(tpl),
            excluded_files=excluded,
        )
        assert "DIFF" in output
        assert "EXCL" in output

    # ---- error handling ----

    def test_nonexistent_template_raises_error(self) -> None:
        bad_path = "/tmp/does_not_exist_xyz.j2"
        with pytest.raises(TemplateNotFoundError) as exc_info:
            self.engine.render(
                _review_result(), _pr_info(), template_path=bad_path
            )
        assert bad_path in str(exc_info.value)

    def test_nonexistent_default_template_raises_error(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            TemplateEngine,
            "DEFAULT_TEMPLATE_PATH",
            "/tmp/no_such_default.j2",
        )
        engine = TemplateEngine()
        with pytest.raises(TemplateNotFoundError):
            engine.render(_review_result(), _pr_info())

    # ---- DEFAULT_TEMPLATE_PATH ----

    def test_default_template_path_value(self) -> None:
        expected = os.path.join("templates", "default.md.j2")
        assert TemplateEngine.DEFAULT_TEMPLATE_PATH == expected


class TestTemplateEngineTokenUsageByGroup:
    """Tests for token_usage_by_group rendering in template."""

    def setup_method(self) -> None:
        self.engine = TemplateEngine()

    def test_render_with_token_usage_by_group(self) -> None:
        """Requirement 11.3: template shows per-group token usage."""
        token_usage = {
            "prompt_tokens": 7000,
            "completion_tokens": 4300,
            "total_tokens": 11300,
        }
        usage_by_group = [
            TokenUsageByGroup("backend", 5000, 3200, 8200),
            TokenUsageByGroup("frontend", 2000, 1100, 3100),
        ]
        output = self.engine.render(
            _review_result(), _pr_info(),
            token_usage=token_usage,
            token_usage_by_group=usage_by_group,
        )
        assert "backend" in output
        assert "8,200" in output
        assert "frontend" in output
        assert "3,100" in output

    def test_render_without_token_usage_by_group(self) -> None:
        """When no group usage, template renders without group section."""
        token_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        output = self.engine.render(
            _review_result(), _pr_info(),
            token_usage=token_usage,
            token_usage_by_group=None,
        )
        # Should still render token usage but no group breakdown
        assert "150" in output

    def test_render_no_token_usage_at_all(self) -> None:
        """When no token usage at all, template renders cleanly."""
        output = self.engine.render(
            _review_result(), _pr_info(),
        )
        # Should not crash, just no token info
        assert "AI" in output  # The footer mentions AI
