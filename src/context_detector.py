"""Context window auto-detection for LLM models."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ContextWindowDetector:
    """Context window 自动探测。"""

    DEFAULT_MAX_CHARS: int = 20_000
    USABLE_RATIO: float = 0.7  # 预留 30% 给 prompt/skills/output

    @staticmethod
    def detect(model: object = None, config_max_chars: int | None = None) -> int:
        """探测优先级：config_max_chars > 模型元数据 > 默认值 20000。

        返回可用于 diff 的最大字符数。
        """
        # 1. Config override takes priority
        if config_max_chars is not None and config_max_chars > 0:
            logger.info("Using config max_chunk_chars: %d (source: config)", config_max_chars)
            return config_max_chars

        # 2. Try model metadata
        if model is not None:
            try:
                # LangChain models expose metadata via .metadata or model attributes
                metadata = getattr(model, "metadata", None) or {}
                context_window = metadata.get("context_window") or metadata.get("max_tokens")
                if context_window and int(context_window) > 0:
                    usable = int(int(context_window) * ContextWindowDetector.USABLE_RATIO)
                    logger.info(
                        "Using model context window: %d * %.0f%% = %d (source: model metadata)",
                        int(context_window),
                        ContextWindowDetector.USABLE_RATIO * 100,
                        usable,
                    )
                    return usable
            except Exception:
                logger.warning("Failed to detect context window from model metadata")

        # 3. Default fallback
        logger.info(
            "Using default max_chunk_chars: %d (source: default)",
            ContextWindowDetector.DEFAULT_MAX_CHARS,
        )
        return ContextWindowDetector.DEFAULT_MAX_CHARS
