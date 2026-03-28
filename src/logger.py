"""Unified logging configuration for the PR Review system."""

from __future__ import annotations

import logging
import os

# Credential env var names whose values must be masked in log output.
_CREDENTIAL_ENV_VARS: list[str] = [
    "GITHUB_TOKEN",
    "GITLAB_TOKEN",
    "CODEUP_TOKEN",
    "LLM_API_KEY",
    "WEBHOOK_SECRET_GITHUB",
    "WEBHOOK_SECRET_GITLAB",
    "WEBHOOK_SECRET_CODEUP",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
]

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


class CredentialMaskingFilter(logging.Filter):
    """Replace credential values with ``***`` in log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.msg)
        if record.args:
            try:
                msg = msg % record.args
            except (TypeError, ValueError):
                pass
            else:
                record.args = None

        for var in _CREDENTIAL_ENV_VARS:
            value = os.environ.get(var)
            if value:
                msg = msg.replace(value, "***")

        record.msg = msg
        return True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with unified format and credential masking."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler.addFilter(CredentialMaskingFilter())
        logger.addHandler(handler)

    return logger
