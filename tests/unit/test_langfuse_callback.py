"""Unit tests for LangfuseCallbackHandler."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, call

import pytest
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.messages import AIMessage

from src.langfuse_callback import LangfuseCallbackHandler


def _make_trace():
    """Create a mock Langfuse trace/span with generation() and span() methods."""
    trace = MagicMock()
    trace.generation.return_value = MagicMock()
    trace.span.return_value = MagicMock()
    return trace


class TestLLMCallbacks:
    """Tests for on_llm_start / on_llm_end / on_llm_error."""

    def test_llm_start_creates_generation(self):
        trace = _make_trace()
        handler = LangfuseCallbackHandler(trace, model_name="gpt-4")
        run_id = uuid.uuid4()

        handler.on_llm_start(
            serialized={},
            prompts=["Hello, world"],
            run_id=run_id,
            invocation_params={"model": "gpt-4", "temperature": 0.7},
        )

        trace.generation.assert_called_once()
        gen_kwargs = trace.generation.call_args
        assert "LLM 第1轮" in gen_kwargs.kwargs["name"]
        assert gen_kwargs.kwargs["model"] == "gpt-4"

    def test_llm_end_closes_generation_with_usage(self):
        trace = _make_trace()
        handler = LangfuseCallbackHandler(trace, model_name="gpt-4")
        run_id = uuid.uuid4()

        # Start
        handler.on_llm_start(
            serialized={}, prompts=["test"], run_id=run_id,
            invocation_params={"model": "gpt-4"},
        )

        # End with token usage
        msg = AIMessage(content="response text")
        msg.usage_metadata = {
            "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
        }
        gen = ChatGeneration(message=msg, text="response text")
        result = LLMResult(
            generations=[[gen]],
            llm_output={"token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }},
        )

        handler.on_llm_end(response=result, run_id=run_id)

        mock_gen = trace.generation.return_value
        mock_gen.end.assert_called_once()
        end_kwargs = mock_gen.end.call_args.kwargs
        assert end_kwargs["usage"]["input"] == 100
        assert end_kwargs["usage"]["output"] == 50
        assert "response text" in end_kwargs["output"]["response"]

    def test_llm_error_records_error(self):
        trace = _make_trace()
        handler = LangfuseCallbackHandler(trace, model_name="gpt-4")
        run_id = uuid.uuid4()

        handler.on_llm_start(
            serialized={}, prompts=["test"], run_id=run_id,
        )
        handler.on_llm_error(
            error=RuntimeError("rate limit"), run_id=run_id,
        )

        mock_gen = trace.generation.return_value
        mock_gen.end.assert_called_once()
        end_kwargs = mock_gen.end.call_args.kwargs
        assert "rate limit" in end_kwargs["output"]["error"]
        assert end_kwargs["level"] == "ERROR"

    def test_llm_turn_counter_increments(self):
        trace = _make_trace()
        handler = LangfuseCallbackHandler(trace, model_name="gpt-4")

        handler.on_llm_start(
            serialized={}, prompts=["p1"], run_id=uuid.uuid4(),
        )
        handler.on_llm_start(
            serialized={}, prompts=["p2"], run_id=uuid.uuid4(),
        )

        calls = trace.generation.call_args_list
        assert "第1轮" in calls[0].kwargs["name"]
        assert "第2轮" in calls[1].kwargs["name"]

    def test_no_trace_does_not_crash(self):
        """Handler with None trace should silently skip."""
        handler = LangfuseCallbackHandler(None, model_name="gpt-4")
        handler.on_llm_start(
            serialized={}, prompts=["test"], run_id=uuid.uuid4(),
        )
        # No exception = pass


class TestToolCallbacks:
    """Tests for on_tool_start / on_tool_end / on_tool_error."""

    def test_tool_start_creates_span(self):
        trace = _make_trace()
        handler = LangfuseCallbackHandler(trace, model_name="gpt-4")
        run_id = uuid.uuid4()

        handler.on_tool_start(
            serialized={"name": "read_file"},
            input_str='{"file_path": "/skills/code-review/SKILL.md"}',
            run_id=run_id,
        )

        trace.span.assert_called_once()
        span_kwargs = trace.span.call_args.kwargs
        assert "read_file" in span_kwargs["name"]

    def test_tool_end_closes_span_with_output(self):
        trace = _make_trace()
        handler = LangfuseCallbackHandler(trace, model_name="gpt-4")
        run_id = uuid.uuid4()

        handler.on_tool_start(
            serialized={"name": "read_file"},
            input_str="test",
            run_id=run_id,
        )
        handler.on_tool_end(output="file content here", run_id=run_id)

        mock_span = trace.span.return_value
        mock_span.end.assert_called_once()
        end_kwargs = mock_span.end.call_args.kwargs
        assert "file content here" in end_kwargs["output"]["result"]

    def test_tool_error_records_error(self):
        trace = _make_trace()
        handler = LangfuseCallbackHandler(trace, model_name="gpt-4")
        run_id = uuid.uuid4()

        handler.on_tool_start(
            serialized={"name": "lookup_symbol"},
            input_str="test",
            run_id=run_id,
        )
        handler.on_tool_error(
            error=FileNotFoundError("not found"), run_id=run_id,
        )

        mock_span = trace.span.return_value
        mock_span.end.assert_called_once()
        end_kwargs = mock_span.end.call_args.kwargs
        assert "not found" in end_kwargs["output"]["error"]

    def test_large_input_truncated(self):
        trace = _make_trace()
        handler = LangfuseCallbackHandler(trace, model_name="gpt-4")

        large_input = "x" * 5000
        handler.on_tool_start(
            serialized={"name": "read_file"},
            input_str=large_input,
            run_id=uuid.uuid4(),
        )

        span_kwargs = trace.span.call_args.kwargs
        raw_input = span_kwargs["input"]["raw_input"]
        assert len(raw_input) < 5000
        assert "chars total" in raw_input
