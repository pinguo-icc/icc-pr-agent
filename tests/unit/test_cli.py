"""Unit tests for CLIRunner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.cli import CLIRunner
from src.exceptions import (
    AllFilesExcludedError,
    EmptyDiffError,
    PRReviewError,
)
from src.models import ReviewOutput, ReviewResult


# ── Argument parsing ──────────────────────────────────────


class TestCLIArgParsing:
    """Verify argparse configuration."""

    def _parse(self, argv: list[str]):
        """Helper: parse args via the internal parser."""
        runner = CLIRunner()
        parser = runner._build_parser()
        return parser.parse_args(argv)

    def test_pr_url_positional(self):
        ns = self._parse(["https://github.com/o/r/pull/1"])
        assert ns.pr_url == "https://github.com/o/r/pull/1"

    def test_template_flag(self):
        ns = self._parse([
            "https://github.com/o/r/pull/1",
            "--template", "my.j2",
        ])
        assert ns.template == "my.j2"

    def test_no_write_back_flag(self):
        ns = self._parse([
            "https://github.com/o/r/pull/1",
            "--no-write-back",
        ])
        assert ns.no_write_back is True

    def test_no_write_back_default(self):
        ns = self._parse(["https://github.com/o/r/pull/1"])
        assert ns.no_write_back is False

    def test_exclude_single(self):
        ns = self._parse([
            "https://github.com/o/r/pull/1",
            "--exclude", "*.lock",
        ])
        assert ns.exclude == ["*.lock"]

    def test_exclude_multiple(self):
        ns = self._parse([
            "https://github.com/o/r/pull/1",
            "--exclude", "*.lock",
            "--exclude", "*.png",
        ])
        assert ns.exclude == ["*.lock", "*.png"]

    def test_exclude_default_is_none(self):
        ns = self._parse(["https://github.com/o/r/pull/1"])
        assert ns.exclude is None

    def test_no_default_exclude_flag(self):
        ns = self._parse([
            "https://github.com/o/r/pull/1",
            "--no-default-exclude",
        ])
        assert ns.no_default_exclude is True

    def test_no_default_exclude_default(self):
        ns = self._parse(["https://github.com/o/r/pull/1"])
        assert ns.no_default_exclude is False


# ── run() behaviour ───────────────────────────────────────


class TestCLIRun:
    """Test CLIRunner.run() with mocked orchestrator."""

    def test_no_args_returns_1(self, capsys):
        rc = CLIRunner().run([])
        assert rc == 1

    @patch("src.cli.ReviewOrchestrator")
    @patch("src.cli.Config")
    def test_happy_path(
        self, mock_config_cls, mock_orch_cls, capsys
    ):
        mock_config_cls.from_env.return_value = MagicMock()
        output = MagicMock(spec=ReviewOutput)
        output.formatted_comment = "# Review OK"
        output.written_back = True
        mock_orch_cls.return_value.run.return_value = output

        rc = CLIRunner().run([
            "https://github.com/o/r/pull/1",
        ])

        assert rc == 0
        captured = capsys.readouterr()
        assert "🔍 开始审查 PR:" in captured.out
        assert "📥 获取 PR 信息..." in captured.out
        assert "✅ 审查完成" in captured.out
        assert "# Review OK" in captured.out
        assert "📤 审查结果已写回 PR" in captured.out

    @patch("src.cli.ReviewOrchestrator")
    @patch("src.cli.Config")
    def test_no_write_back_suppresses_message(
        self, mock_config_cls, mock_orch_cls, capsys
    ):
        mock_config_cls.from_env.return_value = MagicMock()
        output = MagicMock(spec=ReviewOutput)
        output.formatted_comment = "# Review"
        output.written_back = False
        mock_orch_cls.return_value.run.return_value = output

        rc = CLIRunner().run([
            "https://github.com/o/r/pull/1",
            "--no-write-back",
        ])

        assert rc == 0
        captured = capsys.readouterr()
        assert "📤" not in captured.out

    @patch("src.cli.ReviewOrchestrator")
    @patch("src.cli.Config")
    def test_options_forwarded(
        self, mock_config_cls, mock_orch_cls
    ):
        mock_config_cls.from_env.return_value = MagicMock()
        output = MagicMock(spec=ReviewOutput)
        output.formatted_comment = ""
        output.written_back = False
        mock_orch_cls.return_value.run.return_value = output

        CLIRunner().run([
            "https://github.com/o/r/pull/1",
            "--template", "t.j2",
            "--no-write-back",
            "--exclude", "*.lock",
            "--exclude", "*.png",
            "--no-default-exclude",
        ])

        call_args = mock_orch_cls.return_value.run.call_args
        _, opts = call_args[0]
        assert opts.template_path == "t.j2"
        assert opts.write_back is False
        assert opts.exclude_patterns == ["*.lock", "*.png"]
        assert opts.use_default_excludes is False

    @patch("src.cli.ReviewOrchestrator")
    @patch("src.cli.Config")
    def test_empty_diff_error(
        self, mock_config_cls, mock_orch_cls, capsys
    ):
        mock_config_cls.from_env.return_value = MagicMock()
        mock_orch_cls.return_value.run.side_effect = (
            EmptyDiffError("无代码变更")
        )

        rc = CLIRunner().run([
            "https://github.com/o/r/pull/1",
        ])

        assert rc == 1
        captured = capsys.readouterr()
        assert "无代码变更" in captured.err

    @patch("src.cli.ReviewOrchestrator")
    @patch("src.cli.Config")
    def test_all_files_excluded_error(
        self, mock_config_cls, mock_orch_cls, capsys
    ):
        mock_config_cls.from_env.return_value = MagicMock()
        mock_orch_cls.return_value.run.side_effect = (
            AllFilesExcludedError("所有文件被排除")
        )

        rc = CLIRunner().run([
            "https://github.com/o/r/pull/1",
        ])

        assert rc == 1
        captured = capsys.readouterr()
        assert "所有文件被排除" in captured.err

    @patch("src.cli.ReviewOrchestrator")
    @patch("src.cli.Config")
    def test_generic_pr_review_error(
        self, mock_config_cls, mock_orch_cls, capsys
    ):
        mock_config_cls.from_env.return_value = MagicMock()
        mock_orch_cls.return_value.run.side_effect = (
            PRReviewError("something broke")
        )

        rc = CLIRunner().run([
            "https://github.com/o/r/pull/1",
        ])

        assert rc == 1
        captured = capsys.readouterr()
        assert "something broke" in captured.err
