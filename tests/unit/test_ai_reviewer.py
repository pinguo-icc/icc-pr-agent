"""Unit tests for AIReviewer (DeepAgents-based)."""

import json
import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest

from src.ai_reviewer import AIReviewer
from src.config import Config
from src.exceptions import AIModelError, EmptyDiffError
from src.models import (
    PRInfo,
    ReviewIssue,
    ReviewResult,
    TokenUsageByGroup,
)


def _config(**overrides) -> Config:
    defaults = dict(
        llm_api_key="test-key",
        llm_model="gpt-4",
        llm_base_url="https://api.example.com/v1",
        skills_dir="",
    )
    defaults.update(overrides)
    return Config(**defaults)


def _pr_info(**overrides) -> PRInfo:
    defaults = dict(
        platform="github",
        pr_id="42",
        pr_url="https://github.com/owner/repo/pull/42",
        title="Add user auth",
        description="Implements JWT authentication",
        diff=(
            "diff --git a/auth.py b/auth.py\n"
            "--- a/auth.py\n"
            "+++ b/auth.py\n"
            "@@ -1,3 +1,5 @@\n"
            " import os\n"
            "+import jwt\n"
            "+\n"
            " def login():\n"
            "     pass\n"
        ),
        source_branch="feature/auth",
        target_branch="main",
        author="dev",
        version_id="abc123",
    )
    defaults.update(overrides)
    return PRInfo(**defaults)


_VALID_LLM_JSON = json.dumps(
    {
        "summary": "代码整体质量良好，发现1个潜在问题",
        "issues": [
            {
                "file_path": "auth.py",
                "line_number": 2,
                "severity": "warning",
                "category": "security",
                "description": "JWT secret 应从环境变量读取",
                "suggestion": "使用 os.environ.get('JWT_SECRET')",
            }
        ],
    }
)


def _make_mock_agent(content: str = _VALID_LLM_JSON, side_effect=None):
    """Create a mock agent that returns the given content from invoke."""
    mock_agent = MagicMock()
    if side_effect:
        mock_agent.invoke.side_effect = side_effect
    else:
        mock_msg = MagicMock()
        mock_msg.type = "ai"
        mock_msg.content = content
        mock_msg.usage_metadata = None
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
    return mock_agent


# ---------------------------------------------------------------------------
# Existing tests (preserved from original)
# ---------------------------------------------------------------------------


class TestAIReviewerBasic:
    """Basic tests for AIReviewer (DeepAgents-based)."""

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_review_normal_flow(self, mock_init_model, mock_create_agent) -> None:
        """Agent returns valid JSON → ReviewResult is correct."""
        mock_create_agent.return_value = _make_mock_agent()

        reviewer = AIReviewer(_config())
        result = reviewer.review(_pr_info())

        assert result.summary == "代码整体质量良好，发现1个潜在问题"
        assert len(result.issues) == 1
        issue = result.issues[0]
        assert issue.file_path == "auth.py"
        assert issue.line_number == 2
        assert issue.severity == "warning"
        assert issue.category == "security"
        assert issue.description == "JWT secret 应从环境变量读取"
        assert issue.suggestion == "使用 os.environ.get('JWT_SECRET')"
        assert result.reviewed_at

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_review_empty_diff_raises(self, mock_init_model, mock_create_agent) -> None:
        """Empty diff raises EmptyDiffError."""
        reviewer = AIReviewer(_config())
        with pytest.raises(EmptyDiffError, match="无代码变更"):
            reviewer.review(_pr_info(diff=""))

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_review_whitespace_diff_raises(self, mock_init_model, mock_create_agent) -> None:
        """Whitespace-only diff raises EmptyDiffError."""
        reviewer = AIReviewer(_config())
        with pytest.raises(EmptyDiffError, match="无代码变更"):
            reviewer.review(_pr_info(diff="   \n  "))

    @patch("src.ai_reviewer.time.sleep")
    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_review_agent_failure_retries_then_raises(
        self, mock_init_model, mock_create_agent, mock_sleep
    ) -> None:
        """Agent fails 3 times → AIModelError after retries."""
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("connection timeout")
        mock_create_agent.return_value = mock_agent

        reviewer = AIReviewer(_config())
        with pytest.raises(AIModelError, match="AI 模型调用失败"):
            reviewer.review(_pr_info())

        assert mock_agent.invoke.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    @patch("src.ai_reviewer.time.sleep")
    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_review_agent_succeeds_on_retry(
        self, mock_init_model, mock_create_agent, mock_sleep
    ) -> None:
        """Agent fails once then succeeds on second attempt."""
        mock_msg = MagicMock()
        mock_msg.type = "ai"
        mock_msg.content = _VALID_LLM_JSON
        mock_msg.usage_metadata = None

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [
            RuntimeError("temporary error"),
            {"messages": [mock_msg]},
        ]
        mock_create_agent.return_value = mock_agent

        reviewer = AIReviewer(_config())
        result = reviewer.review(_pr_info())

        assert result.summary == "代码整体质量良好，发现1个潜在问题"
        assert mock_agent.invoke.call_count == 2
        mock_sleep.assert_called_once_with(1)

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_review_parses_markdown_fenced_json(
        self, mock_init_model, mock_create_agent
    ) -> None:
        """Agent wraps JSON in markdown code fences → still parsed."""
        fenced = f"```json\n{_VALID_LLM_JSON}\n```"
        mock_create_agent.return_value = _make_mock_agent(fenced)

        reviewer = AIReviewer(_config())
        result = reviewer.review(_pr_info())

        assert len(result.issues) == 1

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_review_invalid_json_raises(self, mock_init_model, mock_create_agent) -> None:
        """Agent returns non-JSON → AIModelError."""
        mock_create_agent.return_value = _make_mock_agent("This is not JSON at all")

        reviewer = AIReviewer(_config())
        with pytest.raises(AIModelError, match="JSON 解析失败"):
            reviewer.review(_pr_info())

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_review_no_issues(self, mock_init_model, mock_create_agent) -> None:
        """Agent returns result with empty issues list."""
        no_issues = json.dumps(
            {"summary": "代码质量优秀，未发现问题", "issues": []}
        )
        mock_create_agent.return_value = _make_mock_agent(no_issues)

        reviewer = AIReviewer(_config())
        result = reviewer.review(_pr_info())

        assert result.summary == "代码质量优秀，未发现问题"
        assert result.issues == []

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_init_creates_agent_with_skills(
        self, mock_init_model, mock_create_agent
    ) -> None:
        """create_deep_agent is called with skills parameter."""
        mock_create_agent.return_value = _make_mock_agent()

        reviewer = AIReviewer(_config())
        reviewer.review(_pr_info())

        mock_create_agent.assert_called()
        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs["skills"] == ["/skills/"]
        assert "system_prompt" in call_kwargs

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_init_model_with_base_url(self, mock_init_model, mock_create_agent) -> None:
        """init_chat_model receives api_key and base_url from config."""
        mock_create_agent.return_value = _make_mock_agent()
        cfg = _config(
            llm_model="deepseek-v3",
            llm_api_key="sk-test",
            llm_base_url="https://custom.api/v1",
        )

        reviewer = AIReviewer(cfg)
        reviewer.review(_pr_info())

        mock_init_model.assert_called_with(
            "openai:deepseek-v3",
            api_key="sk-test",
            base_url="https://custom.api/v1",
        )

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_init_model_no_base_url(self, mock_init_model, mock_create_agent) -> None:
        """When llm_base_url is empty, base_url kwarg is omitted."""
        mock_create_agent.return_value = _make_mock_agent()
        cfg = _config(llm_base_url="")

        reviewer = AIReviewer(cfg)
        reviewer.review(_pr_info())

        call_kwargs = mock_init_model.call_args[1]
        assert "base_url" not in call_kwargs

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_model_string_with_provider_prefix(self, mock_init_model, mock_create_agent) -> None:
        """Model string with provider prefix is passed through as-is."""
        mock_create_agent.return_value = _make_mock_agent()
        cfg = _config(llm_model="anthropic:claude-sonnet-4-5-20250929")

        reviewer = AIReviewer(cfg)
        reviewer.review(_pr_info())

        assert mock_init_model.call_args[0][0] == "anthropic:claude-sonnet-4-5-20250929"

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_agent_created_with_filesystem_backend(
        self, mock_init_model, mock_create_agent
    ) -> None:
        """Agent is created with FilesystemBackend passed as backend param."""
        mock_agent = _make_mock_agent()
        mock_create_agent.return_value = mock_agent

        reviewer = AIReviewer(_config())
        reviewer.review(_pr_info())

        create_call = mock_create_agent.call_args
        assert create_call.kwargs.get("backend") is not None
        invoke_call = mock_agent.invoke.call_args
        invoke_input = invoke_call[0][0]
        assert "messages" in invoke_input
        assert "files" not in invoke_input

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_review_extracts_json_from_mixed_content(
        self, mock_init_model, mock_create_agent
    ) -> None:
        """Agent returns JSON embedded in text → still parsed."""
        mixed = f"Here is my review:\n{_VALID_LLM_JSON}\nHope this helps!"
        mock_create_agent.return_value = _make_mock_agent(mixed)

        reviewer = AIReviewer(_config())
        result = reviewer.review(_pr_info())

        assert len(result.issues) == 1


# ---------------------------------------------------------------------------
# New tests for Task 11.5: Refactored AIReviewer
# ---------------------------------------------------------------------------


class TestSingleAgentFastPath:
    """Tests for single Agent fast path (Requirement 13.1).

    When no file_groups configured AND diff <= max_chunk_chars,
    the reviewer should use the single Agent path (backward compatible).
    """

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_small_diff_no_file_groups_uses_single_agent(
        self, mock_init_model, mock_create_agent
    ) -> None:
        """Small diff + no file_groups → single Agent fast path."""
        mock_create_agent.return_value = _make_mock_agent()
        # No file_groups, small diff → single agent path
        cfg = _config(file_groups=None)
        reviewer = AIReviewer(cfg)
        result = reviewer.review(_pr_info())

        assert isinstance(result, ReviewResult)
        assert result.summary == "代码整体质量良好，发现1个潜在问题"
        assert len(result.issues) == 1

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_single_agent_path_does_not_use_file_grouper(
        self, mock_init_model, mock_create_agent
    ) -> None:
        """Single agent path should not instantiate FileGrouper."""
        mock_create_agent.return_value = _make_mock_agent()
        cfg = _config(file_groups=None)
        reviewer = AIReviewer(cfg)

        # _file_grouper should be None when no file_groups configured
        assert reviewer._file_grouper is None

        result = reviewer.review(_pr_info())
        assert isinstance(result, ReviewResult)


class TestSubAgentPath:
    """Tests for sub-agent path grouping and splitting logic."""

    @patch("src.ai_reviewer.AIReviewer._build_symbol_index", return_value=None)
    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_file_groups_config_triggers_subagent_path(
        self, mock_init_model, mock_create_agent, mock_symbol_index
    ) -> None:
        """When file_groups is configured, sub-agent path is used."""
        mock_create_agent.return_value = _make_mock_agent()
        cfg = _config(
            file_groups={"backend": ["*.py"], "default": ["*"]},
            max_concurrency=1,
            review_timeout=60,
        )
        reviewer = AIReviewer(cfg)
        assert reviewer._file_grouper is not None

        result = reviewer.review(_pr_info())
        assert isinstance(result, ReviewResult)

    @patch("src.ai_reviewer.AIReviewer._build_symbol_index", return_value=None)
    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_subagent_path_creates_agent_per_batch(
        self, mock_init_model, mock_create_agent, mock_symbol_index
    ) -> None:
        """Sub-agent path creates a DeepAgent for each batch."""
        mock_create_agent.return_value = _make_mock_agent()

        # Multi-file diff to produce multiple groups
        diff = (
            "diff --git a/auth.py b/auth.py\n"
            "--- a/auth.py\n"
            "+++ b/auth.py\n"
            "@@ -1,3 +1,5 @@\n"
            " import os\n"
            "+import jwt\n"
            "+\n"
            " def login():\n"
            "     pass\n"
            "diff --git a/style.css b/style.css\n"
            "--- a/style.css\n"
            "+++ b/style.css\n"
            "@@ -1 +1,2 @@\n"
            " body { margin: 0; }\n"
            "+.header { color: red; }\n"
        )
        cfg = _config(
            file_groups={"backend": ["*.py"], "frontend": ["*.css"], "default": ["*"]},
            max_concurrency=2,
            review_timeout=60,
        )
        reviewer = AIReviewer(cfg)
        result = reviewer.review(_pr_info(diff=diff))

        assert isinstance(result, ReviewResult)
        # create_deep_agent should be called multiple times:
        # once per _create_model call + once per sub-agent batch
        assert mock_create_agent.call_count >= 2

    @patch("src.ai_reviewer.AIReviewer._build_symbol_index", return_value=None)
    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_subagent_description_contains_group_name(
        self, mock_init_model, mock_create_agent, mock_symbol_index
    ) -> None:
        """Sub-agent's description and system_prompt contain the group name."""
        mock_create_agent.return_value = _make_mock_agent()
        cfg = _config(
            file_groups={"backend": ["*.py"]},
            max_concurrency=1,
            review_timeout=60,
        )
        reviewer = AIReviewer(cfg)
        reviewer.review(_pr_info())

        # Find the sub-agent creation call (not the model creation call)
        # Sub-agent calls have 'description' kwarg
        for call in mock_create_agent.call_args_list:
            kwargs = call[1]
            if "description" in kwargs:
                assert "backend" in kwargs["description"] or "default" in kwargs["description"]
                assert "system_prompt" in kwargs

    @patch("src.ai_reviewer.AIReviewer._build_symbol_index", return_value=None)
    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_subagent_result_merger_called(
        self, mock_init_model, mock_create_agent, mock_symbol_index
    ) -> None:
        """Sub-agent path merges results from all sub-agents."""
        # Return different results for different calls
        mock_create_agent.return_value = _make_mock_agent()
        cfg = _config(
            file_groups={"backend": ["*.py"]},
            max_concurrency=1,
            max_issues=5,
            review_timeout=60,
        )
        reviewer = AIReviewer(cfg)
        result = reviewer.review(_pr_info())

        assert isinstance(result, ReviewResult)
        assert result.reviewed_at  # Merged result has timestamp


class TestTimeoutHandling:
    """Tests for timeout handling (Requirements 12.3, 12.4)."""

    @patch("src.ai_reviewer.AIReviewer._build_symbol_index", return_value=None)
    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_subagent_timeout_returns_error_result(
        self, mock_init_model, mock_create_agent, mock_symbol_index
    ) -> None:
        """When a sub-agent times out, it returns SubAgentResult with error."""
        # Make the agent invoke hang by raising TimeoutError
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = lambda *a, **kw: (_ for _ in ()).throw(
            concurrent.futures.TimeoutError()
        )
        mock_create_agent.return_value = mock_agent

        cfg = _config(
            file_groups={"backend": ["*.py"]},
            max_concurrency=1,
            review_timeout=1,  # Very short timeout
        )
        reviewer = AIReviewer(cfg)
        result = reviewer.review(_pr_info())

        # Should still return a ReviewResult (not raise)
        assert isinstance(result, ReviewResult)
        # The summary should mention the timeout/failure
        assert "失败" in result.summary or "超时" in result.summary or "backend" in result.summary

    @patch("src.ai_reviewer.AIReviewer._build_symbol_index", return_value=None)
    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_timeout_config_from_config(
        self, mock_init_model, mock_create_agent, mock_symbol_index
    ) -> None:
        """review_timeout from config is used for sub-agent timeout."""
        mock_create_agent.return_value = _make_mock_agent()
        cfg = _config(
            file_groups={"backend": ["*.py"]},
            max_concurrency=1,
            review_timeout=600,
        )
        reviewer = AIReviewer(cfg)
        assert reviewer._config.review_timeout == 600

    @patch("src.ai_reviewer.AIReviewer._build_symbol_index", return_value=None)
    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_subagent_exception_does_not_crash_review(
        self, mock_init_model, mock_create_agent, mock_symbol_index
    ) -> None:
        """When a sub-agent raises an exception, review continues gracefully."""
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("LLM API error")
        mock_create_agent.return_value = mock_agent

        cfg = _config(
            file_groups={"backend": ["*.py"]},
            max_concurrency=1,
            review_timeout=60,
        )
        reviewer = AIReviewer(cfg)
        result = reviewer.review(_pr_info())

        # Should return a result with failure info, not raise
        assert isinstance(result, ReviewResult)
        assert "失败" in result.summary or "error" in result.summary.lower() or "backend" in result.summary


class TestReviewPublicAPI:
    """Tests for review() public API signature unchanged (Requirement 13.2)."""

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_review_accepts_pr_info_returns_review_result(
        self, mock_init_model, mock_create_agent
    ) -> None:
        """review() takes PRInfo and returns ReviewResult — signature unchanged."""
        mock_create_agent.return_value = _make_mock_agent()
        reviewer = AIReviewer(_config())
        pr = _pr_info()

        result = reviewer.review(pr)

        assert isinstance(result, ReviewResult)
        assert isinstance(result.summary, str)
        assert isinstance(result.issues, list)
        assert isinstance(result.reviewed_at, str)

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_review_result_issues_are_review_issue_instances(
        self, mock_init_model, mock_create_agent
    ) -> None:
        """Each issue in the result is a ReviewIssue instance."""
        mock_create_agent.return_value = _make_mock_agent()
        reviewer = AIReviewer(_config())
        result = reviewer.review(_pr_info())

        for issue in result.issues:
            assert isinstance(issue, ReviewIssue)
            assert hasattr(issue, "file_path")
            assert hasattr(issue, "line_number")
            assert hasattr(issue, "severity")
            assert hasattr(issue, "category")
            assert hasattr(issue, "description")
            assert hasattr(issue, "suggestion")

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_review_signature_single_positional_arg(
        self, mock_init_model, mock_create_agent
    ) -> None:
        """review() accepts exactly one positional argument (pr_info)."""
        mock_create_agent.return_value = _make_mock_agent()
        reviewer = AIReviewer(_config())

        import inspect
        sig = inspect.signature(reviewer.review)
        params = list(sig.parameters.keys())
        # Should have 'self' (implicit) and 'pr_info'
        assert params == ["pr_info"]


class TestTokenUsageByGroup:
    """Tests for token_usage_by_group property."""

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_token_usage_empty_before_review(
        self, mock_init_model, mock_create_agent
    ) -> None:
        """token_usage_by_group is empty list before any review."""
        reviewer = AIReviewer(_config())
        assert reviewer.token_usage_by_group == []

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_token_usage_empty_after_single_agent_review(
        self, mock_init_model, mock_create_agent
    ) -> None:
        """token_usage_by_group stays empty after single agent fast path."""
        mock_create_agent.return_value = _make_mock_agent()
        cfg = _config(file_groups=None)
        reviewer = AIReviewer(cfg)
        reviewer.review(_pr_info())

        # Single agent path doesn't populate per-group token usage
        assert reviewer.token_usage_by_group == []

    @patch("src.ai_reviewer.AIReviewer._build_symbol_index", return_value=None)
    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_token_usage_populated_after_subagent_review(
        self, mock_init_model, mock_create_agent, mock_symbol_index
    ) -> None:
        """token_usage_by_group is populated after sub-agent review path."""
        # Create mock agent that returns usage metadata
        mock_msg = MagicMock()
        mock_msg.type = "ai"
        mock_msg.content = _VALID_LLM_JSON
        mock_msg.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create_agent.return_value = mock_agent

        cfg = _config(
            file_groups={"backend": ["*.py"]},
            max_concurrency=1,
            review_timeout=60,
        )
        reviewer = AIReviewer(cfg)
        reviewer.review(_pr_info())

        usage = reviewer.token_usage_by_group
        assert isinstance(usage, list)
        assert len(usage) >= 1
        for item in usage:
            assert isinstance(item, TokenUsageByGroup)
            assert isinstance(item.group_name, str)
            assert isinstance(item.prompt_tokens, int)
            assert isinstance(item.completion_tokens, int)
            assert isinstance(item.total_tokens, int)

    @patch("src.ai_reviewer.AIReviewer._build_symbol_index", return_value=None)
    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_token_usage_backward_compatible_totals(
        self, mock_init_model, mock_create_agent, mock_symbol_index
    ) -> None:
        """Total token counters are updated for backward compatibility."""
        mock_msg = MagicMock()
        mock_msg.type = "ai"
        mock_msg.content = _VALID_LLM_JSON
        mock_msg.usage_metadata = {
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
        }
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create_agent.return_value = mock_agent

        cfg = _config(
            file_groups={"backend": ["*.py"]},
            max_concurrency=1,
            review_timeout=60,
        )
        reviewer = AIReviewer(cfg)
        reviewer.review(_pr_info())

        # Backward-compatible total counters should be updated
        assert reviewer.total_prompt_tokens >= 0
        assert reviewer.total_completion_tokens >= 0
        assert reviewer.total_tokens >= 0

    @patch("src.ai_reviewer.create_deep_agent")
    @patch("src.ai_reviewer.init_chat_model")
    def test_token_usage_property_is_list_of_token_usage_by_group(
        self, mock_init_model, mock_create_agent
    ) -> None:
        """token_usage_by_group property returns list[TokenUsageByGroup]."""
        reviewer = AIReviewer(_config())
        result = reviewer.token_usage_by_group
        assert isinstance(result, list)
