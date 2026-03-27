"""GitHub webhook handler — signature verification & payload parsing."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from src.logger import get_logger

logger = get_logger(__name__)

# PR events that should trigger a review
_REVIEWABLE_ACTIONS = {"opened", "synchronize", "reopened"}


@dataclass
class GitHubWebhookEvent:
    """Parsed GitHub PR webhook event."""

    action: str
    pr_url: str
    pr_number: int
    owner: str
    repo: str
    sender: str
    head_sha: str


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook HMAC-SHA256 signature.

    Args:
        payload: Raw request body bytes.
        signature: Value of ``X-Hub-Signature-256`` header.
        secret: Configured webhook secret.

    Returns:
        ``True`` if the signature is valid.
    """
    if not secret:
        logger.warning("Webhook secret not configured, skipping verification")
        return True

    if not signature or not signature.startswith("sha256="):
        return False

    expected = hmac.new(
        secret.encode(), payload, hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


def parse_pr_event(payload: dict) -> GitHubWebhookEvent | None:
    """Extract PR info from a ``pull_request`` webhook payload.

    Returns:
        A :class:`GitHubWebhookEvent` if the action is reviewable,
        otherwise ``None``.
    """
    action = payload.get("action", "")
    pr = payload.get("pull_request")
    if pr is None:
        logger.debug("Payload has no pull_request field, ignoring")
        return None

    if action not in _REVIEWABLE_ACTIONS:
        logger.info("Ignoring PR action: %s", action)
        return None

    repo_info = payload.get("repository", {})
    owner = repo_info.get("owner", {}).get("login", "")
    repo_name = repo_info.get("name", "")
    pr_number = pr.get("number", 0)
    html_url = pr.get("html_url", "")
    sender = payload.get("sender", {}).get("login", "")
    head_sha = pr.get("head", {}).get("sha", "")

    if not html_url:
        html_url = f"https://github.com/{owner}/{repo_name}/pull/{pr_number}"

    return GitHubWebhookEvent(
        action=action,
        pr_url=html_url,
        pr_number=pr_number,
        owner=owner,
        repo=repo_name,
        sender=sender,
        head_sha=head_sha,
    )
