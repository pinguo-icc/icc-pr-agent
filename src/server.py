"""Unified FastAPI web server for PR Review system.

Mounts:
- /webhook/github  — GitHub webhook callback
- /health          — health check
- (future) /api/*  — management dashboard APIs
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Header, Request

from src.config import Config
from src.exceptions import (
    AllFilesExcludedError,
    EmptyDiffError,
    PRReviewError,
)
from src.logger import get_logger
from src.models import ReviewOptions
from src.orchestrator import ReviewOrchestrator
from src.record_store import RecordStore
from src.langfuse_integration import init_langfuse
from src.webhook.github_handler import (
    GitHubWebhookEvent,
    parse_pr_event,
)

logger = get_logger(__name__)

_config: Config | None = None
_orchestrator: ReviewOrchestrator | None = None
_record_store: RecordStore | None = None

# Serial review queue — one review at a time
_review_queue: asyncio.Queue[GitHubWebhookEvent] = asyncio.Queue()
_worker_task: asyncio.Task | None = None
# Single dedicated thread for blocking review work — avoids default executor
# and prevents "can't start new thread" in resource-constrained containers.
_review_executor: ThreadPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _config, _orchestrator, _record_store, _worker_task, _review_executor
    _config = Config.from_env()
    init_langfuse(_config)
    _orchestrator = ReviewOrchestrator(_config)
    _record_store = RecordStore(_config.review_storage_dir)
    _review_executor = ThreadPoolExecutor(max_workers=1)
    _worker_task = asyncio.create_task(_review_worker())
    logger.info("Server started")
    yield
    # Shutdown: cancel worker, shut down executor
    if _worker_task:
        _worker_task.cancel()
    if _review_executor:
        _review_executor.shutdown(wait=False)
    logger.info("Server shutting down")


app = FastAPI(title="PR Review Server", lifespan=lifespan)


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "queue_size": _review_queue.qsize()}


# ------------------------------------------------------------------
# GitHub Webhook
# ------------------------------------------------------------------

@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    x_github_event: str | None = Header(None),
) -> dict:
    """Handle incoming GitHub webhook events."""
    if x_github_event != "pull_request":
        return {"status": "ignored", "reason": f"event={x_github_event}"}

    payload = await request.json()
    event = parse_pr_event(payload)
    if event is None:
        return {"status": "ignored", "reason": "non-reviewable action"}

    # Dedup: skip if this commit SHA was already reviewed
    assert _record_store is not None
    pr_id = _make_pr_id(event)
    latest = _record_store.get_latest(pr_id)
    if latest and latest.version_id == event.head_sha:
        logger.info(
            "Skipping %s — already reviewed at %s",
            event.pr_url, event.head_sha[:8],
        )
        return {
            "status": "skipped",
            "reason": "already reviewed",
            "version_id": event.head_sha,
        }

    logger.info(
        "Queued PR review: action=%s pr=%s sender=%s (queue=%d)",
        event.action, event.pr_url, event.sender,
        _review_queue.qsize(),
    )
    await _review_queue.put(event)

    return {
        "status": "accepted",
        "pr_url": event.pr_url,
        "action": event.action,
        "queue_position": _review_queue.qsize(),
    }


# ------------------------------------------------------------------
# Serial review worker
# ------------------------------------------------------------------

async def _review_worker() -> None:
    """Process review queue one at a time."""
    logger.info("Review worker started")
    while True:
        event = await _review_queue.get()
        try:
            await _run_review(event)
        except Exception:
            logger.exception("Worker error for %s", event.pr_url)
        finally:
            _review_queue.task_done()


async def _run_review(event: GitHubWebhookEvent) -> None:
    """Execute the review on a dedicated single thread."""
    assert _orchestrator is not None
    assert _review_executor is not None
    options = ReviewOptions(write_back=True)
    loop = asyncio.get_running_loop()

    try:
        output = await loop.run_in_executor(
            _review_executor, _orchestrator.run, event.pr_url, options,
        )
        logger.info(
            "Review done %s — issues=%d tokens=%d written_back=%s",
            event.pr_url,
            len(output.review_result.issues),
            output.total_tokens,
            output.written_back,
        )
    except (EmptyDiffError, AllFilesExcludedError) as exc:
        logger.info("Review skipped for %s: %s", event.pr_url, exc)
    except PRReviewError as exc:
        logger.error("Review failed for %s: %s", event.pr_url, exc)
    except Exception:
        logger.exception("Unexpected error reviewing %s", event.pr_url)


def _make_pr_id(event: GitHubWebhookEvent) -> str:
    """Build pr_id in the format used by RecordStore: owner/repo#number."""
    return f"{event.owner}/{event.repo}#{event.pr_number}"
