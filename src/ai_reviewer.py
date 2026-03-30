"""AI code review engine using LangChain DeepAgents SDK.

Uses ``create_deep_agent`` with skills support for structured code review.
When a diff is too large it is split by file and each chunk is reviewed
independently, then results are merged.

Sub-agent architecture: when ``file_groups`` is configured (or the diff
exceeds the context window), the engine delegates review to per-group
sub-agents running concurrently.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import yaml
from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from src.batch_splitter import BatchSplitter
from src.config import Config
from src.context_detector import ContextWindowDetector
from src.exceptions import AIModelError, EmptyDiffError, SubAgentTimeoutError
from src.file_grouper import FileGrouper
from src.logger import get_logger
from src.models import (
    Batch,
    PRInfo,
    ReviewIssue,
    ReviewResult,
    SubAgentResult,
    TokenUsageByGroup,
)
from src.result_merger import ResultMerger
from src.symbol_indexer import SymbolIndex, SymbolIndexer
from src.langfuse_integration import create_trace, create_span, flush

logger = get_logger(__name__)

# Rough char limit per chunk — leaves room for prompt overhead.
_MAX_DIFF_CHARS = 20_000

_DIFF_HEADER_RE = re.compile(r"^diff --git a/.+ b/.+$", re.MULTILINE)

# ---------- Default prompts (used when no YAML config is provided) ----------

_DEFAULT_SYSTEM_PROMPT = (
    "你是一位资深代码审查专家。请根据你掌握的技能，"
    "对代码变更进行专业审查，严格按照 JSON 格式输出结果。"
)

_DEFAULT_REVIEW_USER_PROMPT = """\
请对以下 Pull Request 的代码变更进行审查。

## PR 信息
- 标题: {title}
- 描述: {description}
- 源分支: {source_branch}
- 目标分支: {target_branch}

## 代码变更 (Diff)
```
{diff}
```

请使用你认为合适的技能进行审查，严格按照以下 JSON 格式输出结果，不要包含其他内容：
{{
  "summary": "审查总结（一段简短的总体评价）",
  "issues": [
    {{
      "file_path": "文件路径",
      "line_number": 行号或null,
      "severity": "critical|warning|suggestion",
      "category": "quality|bug|security|improvement",
      "description": "问题描述",
      "suggestion": "改进建议或null"
    }}
  ]
}}
"""

_DEFAULT_SUMMARY_USER_PROMPT = """\
以下是对一个 Pull Request 中多个文件分别审查后的结果摘要列表。
请将它们合并为一段简洁的总体审查总结（2-3 句话）。

各文件审查摘要：
{summaries}

请只输出总结文本，不要输出 JSON 或其他格式。
"""

_MAX_RETRIES = 3
_BACKOFF_SECONDS = [1, 2, 4]

# Default skills directory (relative to project root)
_DEFAULT_SKILLS_DIR = os.path.join(os.getcwd(), "skills")
_DEFAULT_CONFIG_PATH = os.path.join(os.getcwd(), "pr-review.yaml")


def _load_prompts(config_path: str) -> dict:
    """Load prompt templates from the ``prompts:`` section of pr-review.yaml.

    Returns a dict with keys: system_prompt, review_user_prompt,
    summary_user_prompt.  Missing keys fall back to defaults.
    """
    defaults = {
        "system_prompt": _DEFAULT_SYSTEM_PROMPT,
        "review_user_prompt": _DEFAULT_REVIEW_USER_PROMPT,
        "summary_user_prompt": _DEFAULT_SUMMARY_USER_PROMPT,
    }
    if not config_path or not os.path.isfile(config_path):
        return defaults

    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        prompts_section = data.get("prompts", {})
        if not isinstance(prompts_section, dict):
            return defaults
        loaded = {k: prompts_section[k] for k in defaults if k in prompts_section}
        result = {**defaults, **loaded}
        logger.info(
            "已加载 prompts 配置: %s (覆盖 %d 项)",
            config_path,
            len(loaded),
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 prompts 配置失败: %s，使用默认值", exc)
        return defaults


class AIReviewer:
    """AI-powered code review engine backed by DeepAgents SDK.

    Uses ``create_deep_agent()`` with skills for structured review.
    Supports two paths:
    - Single Agent fast path: small diff with no file_groups config
    - Sub-agent path: delegates to per-group sub-agents for large/grouped diffs
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._skills_dir = config.skills_dir or _DEFAULT_SKILLS_DIR
        config_path = _DEFAULT_CONFIG_PATH
        self._prompts = _load_prompts(config_path)
        # Token usage tracking (backward compatible)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        # Token budget circuit-breaker (0 = unlimited)
        self._token_budget = config.token_budget
        # Sub-agent components
        self._file_grouper = FileGrouper(config.file_groups) if config.file_groups else None
        self._result_merger = ResultMerger()
        # Per-group token usage (populated after sub-agent review)
        self._token_usage_by_group: list[TokenUsageByGroup] = []
        # Trace data for each agent invocation
        self._traces: list[dict] = []
        # Langfuse top-level trace (set in review())
        self._lf_trace = None

    @property
    def token_usage_by_group(self) -> list[TokenUsageByGroup]:
        """Per-group token consumption after sub-agent review."""
        return self._token_usage_by_group

    @property
    def traces(self) -> list[dict]:
        """Trace data from all agent invocations."""
        return self._traces

    @property
    def tools_used(self) -> list[str]:
        """Unique tool names invoked across all traces."""
        names: list[str] = []
        for trace in self._traces:
            for msg in trace.get("messages", []):
                for tc in msg.get("tool_calls", []):
                    name = tc.get("name", "")
                    if name and name not in names:
                        names.append(name)
        return names

    @property
    def skills_loaded(self) -> list[str]:
        """Skill names actually loaded by the agent (extracted from read_file calls)."""
        skills: list[str] = []
        for trace in self._traces:
            for msg in trace.get("messages", []):
                for tc in msg.get("tool_calls", []):
                    if tc.get("name") != "read_file":
                        continue
                    path = tc.get("args", {}).get("file_path", "")
                    # /skills/<skill-name>/SKILL.md
                    if path.startswith("/skills/") and path.endswith("/SKILL.md"):
                        skill = path.split("/")[2]
                        if skill and skill not in skills:
                            skills.append(skill)
        return skills

    def _dump_messages(self, messages: list, group_name: str, batch_index: int) -> None:
        """Build trace data from message chain and store internally."""
        trace = {
            "group_name": group_name,
            "batch_index": batch_index,
            "message_count": len(messages),
            "messages": [],
        }
        for i, msg in enumerate(messages):
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            content_str = str(content)
            tool_calls = []
            # Extract tool calls from ai messages
            if hasattr(msg, "tool_calls"):
                for tc in (msg.tool_calls or []):
                    tool_calls.append({
                        "name": tc.get("name", ""),
                        "args": {k: str(v)[:200] for k, v in tc.get("args", {}).items()},
                    })
            entry = {
                "index": i,
                "role": role,
                "char_count": len(content_str),
                "content_preview": content_str[:300] + ("..." if len(content_str) > 300 else ""),
            }
            if tool_calls:
                entry["tool_calls"] = tool_calls
            # Token usage on last message
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                entry["usage_metadata"] = dict(usage)
            trace["messages"].append(entry)
        self._traces.append(trace)

    def _record_langfuse_messages(self, messages: list, lf_trace) -> None:
        """Record each message in the chain as Langfuse spans/generations.

        Produces a readable trace tree:
        - 📝 用户提问 → span
        - 🤖 LLM 第N轮 → generation (with token usage)
        -   🔧 调用 tool_name(args) → span (nested under the LLM generation)
        -   📎 tool_name 返回结果 → span
        """
        if lf_trace is None:
            return

        llm_turn = 0
        # Map tool_call_id → tool call info for matching tool results
        pending_tool_calls: dict[str, dict] = {}

        for i, msg in enumerate(messages):
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            content_str = str(content)

            if role == "human":
                lf_trace.span(
                    name="📝 用户提问",
                    input={"content": content_str},
                )

            elif role == "ai":
                llm_turn += 1
                tool_calls = []
                if hasattr(msg, "tool_calls"):
                    for tc in (msg.tool_calls or []):
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("args", {})
                        tc_id = tc.get("id", "")
                        tool_calls.append({
                            "name": tc_name,
                            "args": tc_args,
                        })
                        if tc_id:
                            pending_tool_calls[tc_id] = {
                                "name": tc_name,
                                "args": tc_args,
                            }

                usage = getattr(msg, "usage_metadata", None)
                usage_dict = {}
                if usage:
                    usage_dict = {
                        "input": usage.get("input_tokens", 0),
                        "output": usage.get("output_tokens", 0),
                        "total": usage.get("total_tokens", 0),
                    }

                # Build descriptive name
                if tool_calls:
                    tc_names = ", ".join(tc["name"] for tc in tool_calls)
                    gen_name = f"🤖 LLM 第{llm_turn}轮 → 调用 {tc_names}"
                elif content_str.strip():
                    gen_name = f"🤖 LLM 第{llm_turn}轮 → 最终回复"
                else:
                    gen_name = f"🤖 LLM 第{llm_turn}轮"

                gen_output = {}
                if content_str.strip():
                    gen_output["response"] = content_str
                if tool_calls:
                    gen_output["tool_calls"] = tool_calls

                gen = lf_trace.generation(
                    name=gen_name,
                    model=self._config.llm_model,
                    output=gen_output,
                    usage=usage_dict if usage_dict else None,
                    metadata={
                        "turn": llm_turn,
                        "tool_call_count": len(tool_calls),
                    },
                )
                gen.end()

                # Record each tool call as a separate span
                for tc in tool_calls:
                    args_preview = ", ".join(
                        f"{k}={str(v)[:80]}" for k, v in tc["args"].items()
                    )
                    lf_trace.span(
                        name=f"🔧 {tc['name']}({args_preview})",
                        input={"tool_name": tc["name"], "args": tc["args"]},
                    )

            elif role == "tool":
                tool_name = getattr(msg, "name", "unknown")
                tool_call_id = getattr(msg, "tool_call_id", "")

                # Try to get the original call args
                call_info = pending_tool_calls.pop(tool_call_id, None)
                call_args = call_info["args"] if call_info else {}

                # Truncate very long tool results for readability
                result_preview = content_str[:2000]
                if len(content_str) > 2000:
                    result_preview += f"... ({len(content_str)} chars total)"

                lf_trace.span(
                    name=f"📎 {tool_name} 返回结果",
                    input={"tool_name": tool_name, "call_args": call_args},
                    output={"result": result_preview},
                )

    def _build_model_string(self) -> str:
        """Build the provider:model string for init_chat_model."""
        model = self._config.llm_model
        if ":" in model:
            return model
        return f"openai:{model}"

    def _create_model(self):
        """Create a LangChain chat model instance."""
        model_kwargs = {}
        if self._config.llm_api_key:
            model_kwargs["api_key"] = self._config.llm_api_key
        if self._config.llm_base_url:
            model_kwargs["base_url"] = self._config.llm_base_url
        return init_chat_model(self._build_model_string(), **model_kwargs)

    def _create_agent(self):
        """Create a fresh DeepAgents agent instance."""
        model = self._create_model()
        skills_files = self._load_skills_files()

        agent = create_deep_agent(
            model=model,
            system_prompt=self._prompts["system_prompt"],
            skills=["/skills/"],
            checkpointer=MemorySaver(),
        )
        return agent, skills_files

    def _load_skills_files(self) -> dict:
        """Load SKILL.md files from the skills directory into state backend format."""
        skills_files = {}
        skills_dir = self._skills_dir
        if not os.path.isdir(skills_dir):
            logger.warning("Skills 目录不存在: %s", skills_dir)
            return skills_files

        for entry in os.listdir(skills_dir):
            skill_dir = os.path.join(skills_dir, entry)
            if not os.path.isdir(skill_dir):
                continue
            skill_md = os.path.join(skill_dir, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue
            try:
                content = open(skill_md, encoding="utf-8").read()
                virtual_path = f"/skills/{entry}/SKILL.md"
                skills_files[virtual_path] = create_file_data(content)
                logger.info(
                    "已加载 skill: %s (%d chars)",
                    virtual_path,
                    len(content),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("无法读取 skill 文件 %s: %s", skill_md, exc)

        logger.info(
            "Skills 加载完成: %d 个 skill, 来源目录: %s",
            len(skills_files),
            skills_dir,
        )
        return skills_files

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review(self, pr_info: PRInfo) -> ReviewResult:
        """Review PR code changes and return structured results.

        Decision logic:
        - No file_groups config + diff <= max_chunk_chars → single Agent fast path
        - Otherwise → sub-agent path
        """
        if not pr_info.diff or not pr_info.diff.strip():
            raise EmptyDiffError("无代码变更，无需审查")

        diff = pr_info.diff

        # Create a single Langfuse trace for the entire PR review
        # Build trace name: repo#pr_number@time
        # e.g. "order-svc#365@20260328-082519"
        _m = re.search(r'/([^/]+)/pull/(\d+)', pr_info.pr_url)
        _now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        _trace_name = f"{_m.group(1)}#{_m.group(2)}@{_now}" if _m else f"pr-review@{_now}"

        self._lf_trace = create_trace(
            name=_trace_name,
            session_id=pr_info.pr_url,
            metadata={
                "pr_url": pr_info.pr_url,
                "title": pr_info.title,
                "source_branch": pr_info.source_branch,
                "target_branch": pr_info.target_branch,
            },
            tags=["pr-review"],
        )

        # Determine max_chunk_chars for the decision
        model = self._create_model()
        max_chunk_chars = ContextWindowDetector.detect(
            model, self._config.max_chunk_chars,
        )

        # Decision: single Agent fast path vs sub-agent path
        has_file_groups = self._config.file_groups is not None
        if not has_file_groups and len(diff) <= max_chunk_chars:
            # Single Agent fast path (backward compatible)
            result = self._review_single(pr_info, diff)
        elif not has_file_groups and len(diff) > max_chunk_chars:
            # Large diff but no file_groups → use legacy chunked path
            result = self._review_chunked(pr_info)
        else:
            # Sub-agent path
            result = self._review_with_subagents(pr_info, max_chunk_chars)

        # Finalize Langfuse trace
        if self._lf_trace:
            self._lf_trace.update(output={"summary": result.summary})
        flush()
        return result

    # ------------------------------------------------------------------
    # Single-pass review (small diff) — fast path
    # ------------------------------------------------------------------

    def _review_single(self, pr_info: PRInfo, diff: str) -> ReviewResult:
        prompt = self._prompts["review_user_prompt"].format(
            title=pr_info.title,
            description=pr_info.description or "无描述",
            source_branch=pr_info.source_branch,
            target_branch=pr_info.target_branch,
            diff=diff,
        )
        raw = self._call_agent_with_retry(prompt)
        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Chunked review (large diff, no file_groups) — legacy fast path
    # ------------------------------------------------------------------

    def _review_chunked(self, pr_info: PRInfo) -> ReviewResult:
        """Split diff by file, review each chunk, merge results."""
        file_diffs = self._split_diff_by_file(pr_info.diff)
        logger.info(
            "Diff 过大 (%d chars, %d 个文件)，将分片审查",
            len(pr_info.diff),
            len(file_diffs),
        )

        chunks = self._group_into_chunks(file_diffs)
        logger.info("分为 %d 个批次进行审查", len(chunks))

        all_issues: list[ReviewIssue] = []
        summaries: list[str] = []

        for idx, chunk_diff in enumerate(chunks, 1):
            logger.info("审查批次 %d/%d ...", idx, len(chunks))
            result = self._review_single(pr_info, chunk_diff)
            all_issues.extend(result.issues)
            if result.summary:
                summaries.append(result.summary)

            # Circuit-breaker: stop after saving this batch's results
            if self._token_budget > 0 and self.total_tokens >= self._token_budget:
                logger.warning(
                    "Token 预算熔断: 已用 %d, 预算 %d, "
                    "跳过剩余 %d 个批次",
                    self.total_tokens,
                    self._token_budget,
                    len(chunks) - idx,
                )
                summaries.append(
                    f"⚠️ Token 预算熔断（已用 {self.total_tokens}/"
                    f"{self._token_budget}），部分文件未审查。"
                )
                break

        if len(summaries) <= 1:
            merged_summary = summaries[0] if summaries else "审查完成"
        else:
            merged_summary = self._merge_summaries(summaries)

        return ReviewResult(
            summary=merged_summary,
            issues=all_issues,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _split_diff_by_file(diff: str) -> list[str]:
        positions = [m.start() for m in _DIFF_HEADER_RE.finditer(diff)]
        if not positions:
            return [diff]
        sections: list[str] = []
        for i, start in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(diff)
            sections.append(diff[start:end])
        return sections

    @staticmethod
    def _group_into_chunks(
        file_diffs: list[str],
        max_chars: int = _MAX_DIFF_CHARS,
    ) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for fd in file_diffs:
            fd_len = len(fd)
            if current and current_len + fd_len > max_chars:
                chunks.append("".join(current))
                current = []
                current_len = 0
            current.append(fd)
            current_len += fd_len
        if current:
            chunks.append("".join(current))
        return chunks

    def _merge_summaries(self, summaries: list[str]) -> str:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(summaries, 1))
        prompt = self._prompts["summary_user_prompt"].format(
            summaries=numbered,
        )
        try:
            raw = self._call_agent_with_retry(prompt)
            return raw.strip()
        except AIModelError:
            return " | ".join(summaries)

    # ------------------------------------------------------------------
    # Sub-agent review path (Task 11.2)
    # ------------------------------------------------------------------

    def _review_with_subagents(
        self, pr_info: PRInfo, max_chunk_chars: int,
    ) -> ReviewResult:
        """Sub-agent review path.

        1. Group files via FileGrouper
        2. Split each group into batches via BatchSplitter
        3. Optionally build SymbolIndex
        4. Execute sub-agent tasks concurrently
        5. Merge results via ResultMerger
        """
        # File grouping
        grouper = self._file_grouper or FileGrouper()
        file_groups = grouper.group(pr_info.diff)
        logger.info(
            "文件分组完成: %d 个分组 (%s)",
            len(file_groups),
            ", ".join(f"{k}:{len(v.file_paths)}files" for k, v in file_groups.items()),
        )

        # Batch splitting
        splitter = BatchSplitter()
        all_batches: list[Batch] = []
        for group in file_groups.values():
            batches = splitter.split(group, max_chunk_chars)
            all_batches.extend(batches)
        logger.info("共生成 %d 个审查批次", len(all_batches))

        # Optional symbol index
        symbol_index = self._build_symbol_index(pr_info)

        # Resolve repo directory for source file access
        repo_dir = self._resolve_repo_dir(pr_info)

        # Concurrent sub-agent execution
        max_concurrency = self._config.max_concurrency
        sub_results: list[SubAgentResult] = []
        budget_exceeded = False

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(
                    self._create_subagent_task,
                    batch.group_name,
                    batch,
                    symbol_index,
                    pr_info,
                    repo_dir,
                ): batch
                for batch in all_batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    result = future.result()
                    sub_results.append(result)

                    # Accumulate tokens and check budget
                    self.total_prompt_tokens += result.prompt_tokens
                    self.total_completion_tokens += result.completion_tokens
                    self.total_tokens += result.total_tokens

                    if (
                        self._token_budget > 0
                        and self.total_tokens >= self._token_budget
                    ):
                        budget_exceeded = True
                        logger.warning(
                            "Token 预算熔断: 已用 %d, 预算 %d, "
                            "取消剩余 %d 个批次",
                            self.total_tokens,
                            self._token_budget,
                            len(futures) - len(sub_results),
                        )
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break

                except Exception as exc:
                    logger.error(
                        "Sub-agent task failed: group=%s batch=%d error=%s",
                        batch.group_name, batch.batch_index, exc,
                    )
                    sub_results.append(SubAgentResult(
                        group_name=batch.group_name,
                        batch_index=batch.batch_index,
                        result=None,
                        error=str(exc),
                    ))

        # Merge results
        merged = self._result_merger.merge(
            sub_results, max_issues=self._config.max_issues,
        )

        # If budget exceeded, annotate the summary
        if budget_exceeded:
            merged = ReviewResult(
                summary=(
                    f"⚠️ Token 预算熔断（已用 {self.total_tokens}/"
                    f"{self._token_budget}），部分文件未审查。"
                    f"\n\n{merged.summary}"
                ),
                issues=merged.issues,
                reviewed_at=merged.reviewed_at,
            )

        # Aggregate token usage by group
        self._token_usage_by_group = ResultMerger.aggregate_token_usage(sub_results)

        return merged

    def _build_symbol_index(self, pr_info: PRInfo) -> SymbolIndex | None:
        """Optionally build symbol index. Returns None on failure."""
        try:
            indexer = SymbolIndexer(
                cache_dir=self._config.review_storage_dir,
            )
            # Extract repo URL and branch from pr_info
            repo_url = self._extract_repo_url(pr_info)
            if not repo_url:
                logger.warning("无法从 PR 信息中提取仓库 URL，跳过符号索引构建")
                return None

            # Extract changed file paths from diff
            changed_files = self._extract_changed_files(pr_info.diff)

            index = indexer.build(
                repo_url=repo_url,
                branch=pr_info.target_branch,
                changed_files=changed_files,
            )
            logger.info("符号索引构建完成: %d 条记录", len(index.entries))
            return index
        except Exception as exc:
            logger.warning("符号索引构建失败，子 Agent 将不提供 lookup_symbol tool: %s", exc)
            return None

    def _resolve_repo_dir(self, pr_info: PRInfo) -> str | None:
        """Resolve the persistent repo directory for source file access."""
        repo_url = self._extract_repo_url(pr_info)
        if not repo_url:
            return None
        indexer = SymbolIndexer(cache_dir=self._config.review_storage_dir)
        repo_dir = indexer._repo_dir_path(repo_url)
        if os.path.isdir(repo_dir):
            return repo_dir
        return None

    @staticmethod
    def _load_source_files(
        repo_dir: str | None, file_paths: list[str],
    ) -> dict:
        """Load source files from the cloned repo into StateBackend format."""
        source_files = {}
        if not repo_dir:
            return source_files
        for rel_path in file_paths:
            full_path = os.path.join(repo_dir, rel_path)
            if not os.path.isfile(full_path):
                continue
            try:
                content = open(full_path, encoding="utf-8").read()
                # Use the repo-relative path as the virtual path
                virtual_path = f"/{rel_path}"
                source_files[virtual_path] = create_file_data(content)
            except Exception as exc:
                logger.warning("无法读取源文件 %s: %s", full_path, exc)
        if source_files:
            logger.info("已加载 %d 个源文件供 Agent 参考", len(source_files))
        return source_files

    @staticmethod
    def _extract_repo_url(pr_info: PRInfo) -> str | None:
        """Extract clone URL from PR URL."""
        # GitHub: https://github.com/owner/repo/pull/42 → https://github.com/owner/repo.git
        if "github.com" in pr_info.pr_url:
            m = re.match(r"(https://github\.com/[^/]+/[^/]+)", pr_info.pr_url)
            if m:
                return m.group(1) + ".git"
        return None

    @staticmethod
    def _extract_changed_files(diff: str) -> list[str]:
        """Extract file paths from diff headers."""
        paths = []
        for m in re.finditer(r"^diff --git a/.+ b/(.+)$", diff, re.MULTILINE):
            paths.append(m.group(1))
        return paths

    # ------------------------------------------------------------------
    # Sub-agent task creation (Task 11.3)
    # ------------------------------------------------------------------

    def _create_subagent_task(
        self,
        group_name: str,
        batch: Batch,
        symbol_index: SymbolIndex | None,
        pr_info: PRInfo,
        repo_dir: str | None = None,
    ) -> SubAgentResult:
        """Create and execute a single sub-agent review task with timeout protection."""
        start_time = time.monotonic()
        try:
            model = self._create_model()
            skills_files = self._load_skills_files()

            # Load source files for the batch from cloned repo
            source_files = self._load_source_files(repo_dir, batch.file_paths)
            all_files = {**skills_files, **source_files}

            # Build sub-agent system prompt with group context
            sub_system_prompt = (
                f"你是负责审查 {group_name} 分组代码的专家子 Agent。\n"
                f"你正在审查的文件属于 {group_name} 领域。\n"
                f"请先加载 code-review 技能，然后按照技能中的「关联技能加载」要求，"
                f"加载所有与本次审查代码语言相关的技能后，再输出审查结果。\n"
                f"严格按照 code-review 技能定义的 JSON 格式输出结果。\n\n"
                f"如果技能不可用，请从以下维度审查：\n"
                f"1. 代码质量（quality）— 风格、可读性、可维护性\n"
                f"2. 潜在缺陷（bug）— 逻辑错误、边界条件、资源泄漏\n"
                f"3. 安全风险（security）— 注入、信息泄露、权限问题\n"
                f"4. 改进建议（improvement）— 性能优化、更好的实现方式"
            )

            # Build tools for sub-agent
            tools = []
            if symbol_index is not None:
                lookup_tool = self._build_lookup_symbol_tool(symbol_index)
                tools.append(lookup_tool)

            # Create sub-agent using DeepAgents SDK
            sub_agent = create_deep_agent(
                model=model,
                system_prompt=sub_system_prompt,
                name=f"review-{group_name}",
                skills=["/skills/"],
                tools=tools,
                checkpointer=MemorySaver(),
            )

            # Build review prompt
            prompt = self._prompts["review_user_prompt"].format(
                title=pr_info.title,
                description=pr_info.description or "无描述",
                source_branch=pr_info.source_branch,
                target_branch=pr_info.target_branch,
                diff=batch.diff_content,
            )

            thread_id = (
                f"review-{group_name}-{batch.batch_index}-"
                f"{datetime.now(timezone.utc).timestamp()}"
            )

            # Langfuse span under the top-level PR trace
            trace = create_span(
                self._lf_trace,
                name=f"subagent-{group_name}-batch{batch.batch_index}",
                metadata={
                    "group_name": group_name,
                    "batch_index": batch.batch_index,
                },
            )
            if trace:
                trace.update(input={"prompt": prompt})

            # Execute directly (no nested thread pool — thread budget is managed
            # by the outer ThreadPoolExecutor in _review_with_subagents).
            # Retry on 429 rate-limit errors with exponential back-off.
            last_invoke_error: Exception | None = None
            result = None
            for _attempt in range(_MAX_RETRIES):
                try:
                    result = sub_agent.invoke(
                        {
                            "messages": [{"role": "user", "content": prompt}],
                            "files": all_files,
                        },
                        config={
                            "configurable": {"thread_id": thread_id},
                        },
                    )
                    last_invoke_error = None
                    break
                except Exception as invoke_exc:
                    last_invoke_error = invoke_exc
                    if "429" in str(invoke_exc) or "rate limit" in str(invoke_exc).lower():
                        if _attempt < _MAX_RETRIES - 1:
                            wait = _BACKOFF_SECONDS[_attempt]
                            logger.warning(
                                "Sub-agent 429 rate limit, %ds 后重试 "
                                "(attempt %d/%d): group=%s batch=%d",
                                wait, _attempt + 1, _MAX_RETRIES,
                                group_name, batch.batch_index,
                            )
                            time.sleep(wait)
                            continue
                    # Non-retryable error, raise immediately
                    raise

            if last_invoke_error is not None:
                raise last_invoke_error

            # Parse response
            messages = result.get("messages", [])
            if not messages:
                raise AIModelError("Sub-agent 返回空消息")

            # Dump full message chain for token analysis
            self._dump_messages(messages, group_name, batch.batch_index)

            last_msg = messages[-1]
            content = (
                last_msg.content
                if hasattr(last_msg, "content")
                else str(last_msg)
            )

            # Track token usage
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens_val = 0
            usage = getattr(last_msg, "usage_metadata", None)
            if usage:
                prompt_tokens = usage.get("input_tokens", 0)
                completion_tokens = usage.get("output_tokens", 0)
                total_tokens_val = usage.get("total_tokens", 0)

            elapsed = time.monotonic() - start_time
            review_result = self._parse_response(content)

            # End Langfuse: record each message as span/generation
            if trace:
                self._record_langfuse_messages(messages, trace)
                trace.update(output={"content": content})
            return SubAgentResult(
                group_name=group_name,
                batch_index=batch.batch_index,
                result=review_result,
                error=None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens_val,
                elapsed_seconds=elapsed,
            )

        except SubAgentTimeoutError:
            raise
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            logger.error(
                "Sub-agent 审查异常: group=%s batch=%d error=%s",
                group_name, batch.batch_index, exc,
            )
            return SubAgentResult(
                group_name=group_name,
                batch_index=batch.batch_index,
                result=None,
                error=str(exc),
                elapsed_seconds=elapsed,
            )

    # ------------------------------------------------------------------
    # lookup_symbol tool builder (Task 11.4)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_lookup_symbol_tool(symbol_index: SymbolIndex):
        """Build a lookup_symbol tool for sub-agents.

        The tool queries the SymbolIndex by symbol name and optional file_hint.
        Returns symbol signature, file path, and line number when found.
        Returns "symbol not found in project" when not found.
        """

        @tool
        def lookup_symbol(name: str, file_hint: str = "") -> str:
            """查询项目内部符号的签名信息。

            Args:
                name: 符号名称（函数名、类名、方法名等）
                file_hint: 可选的文件路径提示，用于缩小查询范围

            Returns:
                符号的签名、文件路径和行号信息，或"符号未在项目内部找到"提示
            """
            hint = file_hint if file_hint else None
            entries = symbol_index.lookup(name, file_hint=hint)
            if not entries:
                return f"符号 '{name}' 未在项目内部找到（可能是第三方库的符号）"

            results = []
            for entry in entries:
                results.append(
                    f"- {entry.kind} {entry.name}: {entry.signature}\n"
                    f"  文件: {entry.file_path}, 行号: {entry.line_number}, "
                    f"语言: {entry.language}"
                )
            return "\n".join(results)

        return lookup_symbol

    # ------------------------------------------------------------------
    # DeepAgents invocation with retry
    # ------------------------------------------------------------------

    def _call_agent_with_retry(self, prompt: str) -> str:
        """Invoke the DeepAgents agent with up to 3 retries."""
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                logger.info(
                    "调用 DeepAgents (尝试 %d/%d)", attempt + 1, _MAX_RETRIES
                )
                agent, skills_files = self._create_agent()
                thread_id = f"review-{datetime.now(timezone.utc).timestamp()}"

                # Langfuse span under the top-level PR trace
                trace = create_span(
                    self._lf_trace,
                    name=f"single-agent-attempt{attempt + 1}",
                    metadata={"attempt": attempt + 1},
                )
                if trace:
                    trace.update(input={"prompt": prompt})

                result = agent.invoke(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "files": skills_files,
                    },
                    config={
                        "configurable": {"thread_id": thread_id},
                    },
                )
                # Extract the final assistant message
                messages = result.get("messages", [])
                if not messages:
                    raise AIModelError("DeepAgents 返回空消息")

                # Dump full message chain for token analysis
                self._dump_messages(messages, "single", 0)

                last_msg = messages[-1]
                content = (
                    last_msg.content
                    if hasattr(last_msg, "content")
                    else str(last_msg)
                )

                # Track token usage from response metadata if available
                usage = getattr(last_msg, "usage_metadata", None)
                if usage:
                    self.total_prompt_tokens += usage.get("input_tokens", 0)
                    self.total_completion_tokens += usage.get("output_tokens", 0)
                    self.total_tokens += usage.get("total_tokens", 0)

                # End Langfuse generation with token usage
                if trace:
                    self._record_langfuse_messages(messages, trace)
                    trace.update(output={"content": content})
                return content
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = _BACKOFF_SECONDS[attempt]
                    logger.warning(
                        "DeepAgents 调用失败，%ds 后重试: %s", wait, exc
                    )
                    time.sleep(wait)

        raise AIModelError(f"AI 模型调用失败: {last_error}")

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> ReviewResult:
        """Parse the agent JSON response into a ReviewResult."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Try to extract JSON from mixed content
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AIModelError(
                f"AI 返回的 JSON 解析失败: {exc}"
            ) from exc

        issues = [
            ReviewIssue(
                file_path=item.get("file_path", ""),
                line_number=item.get("line_number"),
                severity=item.get("severity", "suggestion"),
                category=item.get("category", "improvement"),
                description=item.get("description", ""),
                suggestion=item.get("suggestion"),
                example=item.get("example"),
                old_code=item.get("old_code"),
            )
            for item in data.get("issues", [])
        ]

        return ReviewResult(
            summary=data.get("summary", ""),
            issues=issues,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )
