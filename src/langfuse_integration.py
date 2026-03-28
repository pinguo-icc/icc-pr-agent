"""Langfuse observability integration for the PR Review system.

Uses the low-level Langfuse SDK (trace/generation) instead of the LangChain
CallbackHandler to avoid version compatibility issues between langfuse SDK,
LangChain, and self-hosted Langfuse server.

Usage in ai_reviewer.py::

    from src.langfuse_integration import get_langfuse, create_trace

    trace = create_trace(name="pr-review", metadata={...})
    if trace:
        gen = trace.generation(name="llm-call", input=prompt)
        # ... do LLM call ...
        gen.end(output=content, usage={"input": 100, "output": 50})
        trace.update(output={"summary": "..."})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.logger import get_logger

if TYPE_CHECKING:
    from src.config import Config

logger = get_logger(__name__)

_langfuse_client = None
_initialized = False


def init_langfuse(config: Config) -> None:
    """Initialize the global Langfuse client. Call once at startup."""
    global _langfuse_client, _initialized
    if _initialized:
        return

    _initialized = True

    if not config.langfuse_enabled:
        logger.info("Langfuse 已禁用 (LANGFUSE_ENABLED=false)")
        return

    if not config.langfuse_public_key or not config.langfuse_secret_key:
        logger.info("Langfuse 未配置 (缺少 LANGFUSE_PUBLIC_KEY 或 LANGFUSE_SECRET_KEY)，跳过")
        return

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
        )
        _langfuse_client.auth_check()
        logger.info("Langfuse 初始化成功: host=%s", config.langfuse_host)
    except ImportError:
        logger.warning("langfuse 未安装，跳过 observability 集成")
    except Exception as exc:
        logger.warning("Langfuse 初始化失败: %s", exc)
        _langfuse_client = None


def get_langfuse():
    """Return the global Langfuse client, or None."""
    return _langfuse_client


def create_trace(
    *,
    name: str = "pr-review",
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    """Create a new Langfuse trace. Returns None if not available."""
    if _langfuse_client is None:
        return None
    try:
        return _langfuse_client.trace(
            name=name,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
            tags=tags or [],
        )
    except Exception as exc:
        logger.warning("Langfuse trace 创建失败: %s", exc)
        return None


def flush() -> None:
    """Flush pending Langfuse events."""
    if _langfuse_client is None:
        return
    try:
        _langfuse_client.flush()
    except Exception as exc:
        logger.warning("Langfuse flush 失败: %s", exc)
