"""Environment-based configuration for the PR Review system."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_dotenv(env_path: str = ".env") -> None:
    """Load key=value pairs from a .env file into os.environ (if file exists)."""
    path = Path(env_path)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not os.environ.get(key):
            os.environ[key] = value


def _load_yaml_config(yaml_path: str = "pr-review.yaml") -> dict[str, Any]:
    """Load configuration from a YAML file. Returns empty dict on failure."""
    path = Path(yaml_path)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("Failed to load YAML config from %s", yaml_path)
        return {}


def _parse_positive_int(
    value: Any, field_name: str, default: int,
) -> int:
    """Parse a value as a positive integer. Returns default on failure."""
    try:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("not positive")
        return parsed
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value '%s' for %s, using default %d",
            value, field_name, default,
        )
        return default


@dataclass
class Config:
    """Lazily loaded configuration backed by environment variables."""

    # Platform tokens
    github_token: str = ""
    gitlab_token: str = ""
    gitlab_url: str = "https://gitlab.com"
    codeup_token: str = ""
    codeup_org_id: str = ""

    # LLM settings
    llm_api_key: str = ""
    llm_model: str = "gpt-4"
    llm_base_url: str = ""

    # General settings
    log_level: str = "INFO"
    review_storage_dir: str = ".pr_reviews"
    pr_review_exclude: list[str] = field(default_factory=list)
    skills_dir: str = ""

    # Webhook secrets
    webhook_secret_github: str = ""
    webhook_secret_gitlab: str = ""
    webhook_secret_codeup: str = ""

    # Sub-agent review settings
    file_groups: dict[str, list[str]] | None = None
    max_chunk_chars: int | None = None
    max_issues: int = 10
    max_concurrency: int = 3
    review_timeout: int = 300

    # Token budget (0 = unlimited)
    token_budget: int = 0

    # Langfuse observability
    langfuse_enabled: bool = True
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    @classmethod
    def from_env(
        cls, dotenv_path: str = ".env", yaml_path: str = "pr-review.yaml",
    ) -> Config:
        """Create a Config instance from environment variables and YAML config.

        Loads .env file first (won't override existing env vars).
        Priority: environment variable > YAML config > default value.
        """
        _load_dotenv(dotenv_path)
        yaml_cfg = _load_yaml_config(yaml_path)

        exclude_raw = os.environ.get("PR_REVIEW_EXCLUDE", "")
        exclude_patterns = [
            p.strip() for p in exclude_raw.split(",") if p.strip()
        ]

        # --- file_groups: YAML only (no env var) ---
        file_groups = yaml_cfg.get("file_groups")
        if file_groups is not None and not isinstance(file_groups, dict):
            logger.warning(
                "Invalid file_groups in YAML config, ignoring"
            )
            file_groups = None

        # --- max_chunk_chars: YAML only (no env var) ---
        max_chunk_chars: int | None = None
        yaml_max_chunk = yaml_cfg.get("max_chunk_chars")
        if yaml_max_chunk is not None:
            try:
                parsed = int(yaml_max_chunk)
                if parsed <= 0:
                    raise ValueError("not positive")
                max_chunk_chars = parsed
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid value '%s' for max_chunk_chars, ignoring",
                    yaml_max_chunk,
                )

        # --- max_issues: env > YAML > default(10) ---
        default_max_issues = 10
        env_max_issues = os.environ.get("MAX_REVIEW_ISSUES")
        if env_max_issues is not None:
            max_issues = _parse_positive_int(
                env_max_issues, "max_issues", default_max_issues,
            )
        elif "max_issues" in yaml_cfg:
            max_issues = _parse_positive_int(
                yaml_cfg["max_issues"], "max_issues", default_max_issues,
            )
        else:
            max_issues = default_max_issues

        # --- max_concurrency: env > YAML > default(3) ---
        default_max_concurrency = 3
        env_max_concurrency = os.environ.get("MAX_REVIEW_CONCURRENCY")
        if env_max_concurrency is not None:
            max_concurrency = _parse_positive_int(
                env_max_concurrency, "max_concurrency",
                default_max_concurrency,
            )
        elif "max_concurrency" in yaml_cfg:
            max_concurrency = _parse_positive_int(
                yaml_cfg["max_concurrency"], "max_concurrency",
                default_max_concurrency,
            )
        else:
            max_concurrency = default_max_concurrency

        # --- review_timeout: env > YAML > default(300) ---
        default_review_timeout = 300
        env_review_timeout = os.environ.get("REVIEW_TIMEOUT")
        if env_review_timeout is not None:
            review_timeout = _parse_positive_int(
                env_review_timeout, "review_timeout",
                default_review_timeout,
            )
        elif "review_timeout" in yaml_cfg:
            review_timeout = _parse_positive_int(
                yaml_cfg["review_timeout"], "review_timeout",
                default_review_timeout,
            )
        else:
            review_timeout = default_review_timeout

        # --- token_budget: env > YAML > default(0 = unlimited) ---
        default_token_budget = 0
        env_token_budget = os.environ.get("TOKEN_BUDGET")
        if env_token_budget is not None:
            try:
                token_budget = int(env_token_budget)
                if token_budget < 0:
                    raise ValueError("negative")
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid value '%s' for TOKEN_BUDGET, using default %d",
                    env_token_budget, default_token_budget,
                )
                token_budget = default_token_budget
        elif "token_budget" in yaml_cfg:
            try:
                token_budget = int(yaml_cfg["token_budget"])
                if token_budget < 0:
                    raise ValueError("negative")
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid value '%s' for token_budget, using default %d",
                    yaml_cfg["token_budget"], default_token_budget,
                )
                token_budget = default_token_budget
        else:
            token_budget = default_token_budget

        return cls(
            langfuse_enabled=os.environ.get(
                "LANGFUSE_ENABLED", "true"
            ).lower() in ("true", "1", "yes"),
            langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            langfuse_host=os.environ.get(
                "LANGFUSE_HOST", "http://localhost:3000"
            ),
            github_token=os.environ.get("GITHUB_TOKEN", ""),
            gitlab_token=os.environ.get("GITLAB_TOKEN", ""),
            gitlab_url=os.environ.get("GITLAB_URL", "https://gitlab.com"),
            codeup_token=os.environ.get("CODEUP_TOKEN", ""),
            codeup_org_id=os.environ.get("CODEUP_ORG_ID", ""),
            llm_api_key=os.environ.get("LLM_API_KEY", ""),
            llm_model=os.environ.get("LLM_MODEL", "gpt-4"),
            llm_base_url=os.environ.get("LLM_BASE_URL", ""),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            review_storage_dir=os.environ.get(
                "REVIEW_STORAGE_DIR", ".pr_reviews"
            ),
            pr_review_exclude=exclude_patterns,
            skills_dir=os.environ.get("SKILLS_DIR", ""),
            webhook_secret_github=os.environ.get("WEBHOOK_SECRET_GITHUB", ""),
            webhook_secret_gitlab=os.environ.get("WEBHOOK_SECRET_GITLAB", ""),
            webhook_secret_codeup=os.environ.get(
                "WEBHOOK_SECRET_CODEUP", ""
            ),
            file_groups=file_groups,
            max_chunk_chars=max_chunk_chars,
            max_issues=max_issues,
            max_concurrency=max_concurrency,
            review_timeout=review_timeout,
            token_budget=token_budget,
        )
