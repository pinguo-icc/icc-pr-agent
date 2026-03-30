"""LangChain Callback Handler for real-time Langfuse tracing.

Intercepts every LLM call and tool call at the LangChain runtime level,
recording raw request/response data to Langfuse in real-time (instead of
post-hoc message chain replay).

Usage::

    handler = LangfuseCallbackHandler(parent_trace, model_name="gpt-4")
    result = agent.invoke(input, config={"callbacks": [handler]})

Replaces the old ``_record_langfuse_messages()`` approach.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.logger import get_logger

logger = get_logger(__name__)


class LangfuseCallbackHandler(BaseCallbackHandler):
    """Real-time LLM & tool call tracing via Langfuse low-level SDK."""

    def __init__(self, parent_trace, *, model_name: str = "unknown"):
        """
        Args:
            parent_trace: Langfuse trace or span to nest observations under.
            model_name: LLM model identifier for generation records.
        """
        super().__init__()
        self._trace = parent_trace
        self._model_name = model_name
        self._llm_turn = 0
        # run_id → (generation_object, start_time)
        self._generations: dict[str, tuple[Any, float]] = {}
        # run_id → (span_object, start_time)
        self._tool_spans: dict[str, tuple[Any, float]] = {}

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts. Captures raw input messages/prompts."""
        if self._trace is None:
            return
        try:
            self._llm_turn += 1
            invocation_params = kwargs.get("invocation_params", {})

            # Build a readable input snapshot
            input_data: dict[str, Any] = {}
            if prompts:
                # For chat models, prompts is the serialized message list
                input_data["prompts"] = [
                    p[:2000] + ("..." if len(p) > 2000 else "") for p in prompts
                ]
            if invocation_params:
                # Capture model params: temperature, max_tokens, etc.
                safe_params = {
                    k: v for k, v in invocation_params.items()
                    if k in (
                        "model", "model_name", "temperature", "max_tokens",
                        "top_p", "stop", "stream",
                    )
                }
                if safe_params:
                    input_data["model_params"] = safe_params

            gen = self._trace.generation(
                name=f"🤖 LLM 第{self._llm_turn}轮",
                model=invocation_params.get("model", self._model_name),
                input=input_data,
                metadata={
                    "turn": self._llm_turn,
                    "run_id": str(run_id),
                },
            )
            self._generations[str(run_id)] = (gen, time.monotonic())

            # Log LLM input summary for debugging
            total_prompt_chars = sum(len(p) for p in prompts) if prompts else 0
            logger.info(
                "LLM 第%d轮开始: model=%s prompt_count=%d total_chars=%d",
                self._llm_turn,
                invocation_params.get("model", self._model_name),
                len(prompts) if prompts else 0,
                total_prompt_chars,
            )
        except Exception as exc:
            logger.debug("Langfuse on_llm_start 记录失败: %s", exc)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Called when LLM finishes. Captures raw output + token usage."""
        entry = self._generations.pop(str(run_id), None)
        if entry is None:
            return
        gen, start_time = entry
        try:
            elapsed_ms = (time.monotonic() - start_time) * 1000

            # Extract response content
            output_data: dict[str, Any] = {}
            if response.generations:
                for gen_list in response.generations:
                    for g in gen_list:
                        text = g.text or ""
                        output_data["response"] = (
                            text[:3000] + ("..." if len(text) > 3000 else "")
                        )
                        # Additional info from generation_info
                        if g.generation_info:
                            output_data["generation_info"] = {
                                k: v for k, v in g.generation_info.items()
                                if k in ("finish_reason", "logprobs")
                            }
                        # Tool calls from AIMessage
                        msg = g.message if hasattr(g, "message") else None
                        if msg and hasattr(msg, "tool_calls") and msg.tool_calls:
                            output_data["tool_calls"] = [
                                {"name": tc.get("name", ""), "args": tc.get("args", {})}
                                for tc in msg.tool_calls
                            ]
                        break  # first generation only
                    break

            # Extract token usage
            usage: dict[str, int] = {}
            if response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                if token_usage:
                    usage = {
                        "input": token_usage.get("prompt_tokens", 0),
                        "output": token_usage.get("completion_tokens", 0),
                        "total": token_usage.get("total_tokens", 0),
                    }
            # Fallback: try usage_metadata on the message object
            if not usage and response.generations:
                for gen_list in response.generations:
                    for g in gen_list:
                        msg = g.message if hasattr(g, "message") else None
                        if msg:
                            um = getattr(msg, "usage_metadata", None)
                            if um:
                                usage = {
                                    "input": um.get("input_tokens", 0),
                                    "output": um.get("output_tokens", 0),
                                    "total": um.get("total_tokens", 0),
                                }
                        break
                    break

            output_data["elapsed_ms"] = round(elapsed_ms, 1)

            gen.end(
                output=output_data,
                usage=usage if usage else None,
            )

            # Log LLM response summary
            response_len = len(output_data.get("response", ""))
            tc_count = len(output_data.get("tool_calls", []))
            logger.info(
                "LLM 第%d轮结束: response_len=%d tool_calls=%d "
                "tokens=%s elapsed=%.0fms",
                self._llm_turn, response_len, tc_count,
                usage if usage else "N/A", elapsed_ms,
            )
        except Exception as exc:
            logger.debug("Langfuse on_llm_end 记录失败: %s", exc)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors. Records the error on the generation."""
        entry = self._generations.pop(str(run_id), None)
        if entry is None:
            return
        gen, _ = entry
        try:
            gen.end(
                output={"error": str(error)[:1000]},
                level="ERROR",
            )
        except Exception as exc:
            logger.debug("Langfuse on_llm_error 记录失败: %s", exc)

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts. Captures tool name + raw input."""
        if self._trace is None:
            return
        try:
            tool_name = serialized.get("name", "unknown_tool")
            # input_str can be very large (e.g. file content), truncate
            input_preview = input_str[:3000]
            if len(input_str) > 3000:
                input_preview += f"... ({len(input_str)} chars total)"

            span = self._trace.span(
                name=f"🔧 {tool_name}",
                input={"tool_name": tool_name, "raw_input": input_preview},
                metadata={"run_id": str(run_id)},
            )
            self._tool_spans[str(run_id)] = (span, time.monotonic())
        except Exception as exc:
            logger.debug("Langfuse on_tool_start 记录失败: %s", exc)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes. Captures raw output."""
        entry = self._tool_spans.pop(str(run_id), None)
        if entry is None:
            return
        span, start_time = entry
        try:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            output_str = str(output)
            output_preview = output_str[:3000]
            if len(output_str) > 3000:
                output_preview += f"... ({len(output_str)} chars total)"

            span.end(
                output={"result": output_preview, "elapsed_ms": round(elapsed_ms, 1)},
            )
        except Exception as exc:
            logger.debug("Langfuse on_tool_end 记录失败: %s", exc)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        entry = self._tool_spans.pop(str(run_id), None)
        if entry is None:
            return
        span, _ = entry
        try:
            span.end(
                output={"error": str(error)[:1000]},
            )
        except Exception as exc:
            logger.debug("Langfuse on_tool_error 记录失败: %s", exc)
