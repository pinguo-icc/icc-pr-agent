"""Property-based tests for AIReviewer sub-agent metadata."""

import json
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from src.ai_reviewer import AIReviewer
from src.config import Config
from src.models import Batch


_VALID_LLM_JSON = json.dumps({
    "summary": "ok",
    "issues": [],
})


# --- Property 4: 子 Agent 元数据包含分组信息 ---
# Feature: sub-agent-review, Property 4: 子 Agent 元数据包含分组信息

@given(group_name=st.from_regex(r"[a-z]{2,10}", fullmatch=True))
@settings(max_examples=100)
def test_property_4_subagent_metadata_contains_group(group_name):
    """Sub-agent description and system_prompt contain the group name.

    Validates: Requirements 3.1, 3.3, 3.4
    """
    with patch("src.ai_reviewer.create_deep_agent") as mock_create, \
         patch("src.ai_reviewer.init_chat_model"):

        mock_msg = MagicMock()
        mock_msg.content = _VALID_LLM_JSON
        mock_msg.usage_metadata = None
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create.return_value = mock_agent

        cfg = Config(
            llm_api_key="test", llm_model="gpt-4",
            file_groups={group_name: ["*.py"]},
            max_concurrency=1, review_timeout=60,
        )
        reviewer = AIReviewer(cfg)

        batch = Batch(
            group_name=group_name, batch_index=0,
            file_paths=["test.py"], diff_content="diff content",
            char_count=12,
        )

        from src.models import PRInfo
        pr = PRInfo(
            platform="github", pr_id="1",
            pr_url="https://github.com/o/r/pull/1",
            title="t", description="d", diff="diff",
            source_branch="f", target_branch="main",
            author="a", version_id="v",
        )

        reviewer._create_subagent_task(group_name, batch, None, pr)

        # Find the sub-agent creation call with description
        for call in mock_create.call_args_list:
            kwargs = call[1]
            if "description" in kwargs:
                assert group_name in kwargs["description"]
                assert group_name in kwargs["system_prompt"]
                return

        # If no call with description found, that's a failure
        assert False, "create_deep_agent was not called with description"
