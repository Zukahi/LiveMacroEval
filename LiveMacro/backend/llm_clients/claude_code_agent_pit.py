"""
Claude Code Agent restricted to point-in-time search.

Same signed-in Claude Code path as claude_code_agent.py, with one decisive change:
the built-in WebSearch and WebFetch are DENIED, and the agent's only window on the
world is a pair of in-process MCP tools that route through search_pit and refuse to
return anything published on or after the cutoff.

That turns the evidence mode's leakage audit from a detector into a guarantee. In a
backtest the agent cannot read the answer, because the tool that would have to hand
it over will not.

The cutoff is bound per run by build_pit_client(as_of), not taken from the model's
arguments — otherwise the agent could widen its own window.
"""

import asyncio
import os
from collections import deque
from contextlib import suppress

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, create_sdk_mcp_server, query, tool
from claude_agent_sdk.types import ToolResultBlock, ToolUseBlock

from config import get_logger
from search_pit import PitSearchError, pit_get_contents, pit_search

from .claude_code_agent import (
    MODEL_ID,
    _LIVEMACRO_ROOT,
    _build_stderr_collector,
    _get_agent_timeout_secs,
    _signed_in_sdk_only_env,
)

logger = get_logger(__name__)

EFFORT_LEVEL = "high"
DISPLAY_NAME = (
    "Claude Code Agent, point-in-time "
    "(claude-sonnet-4-6, effort=high, date-filtered search only — no WebSearch/WebFetch)"
)

MCP_SERVER_NAME = "pit"
SEARCH_TOOL = f"mcp__{MCP_SERVER_NAME}__search"
CONTENTS_TOOL = f"mcp__{MCP_SERVER_NAME}__get_contents"
ALLOWED_TOOLS = [SEARCH_TOOL, CONTENTS_TOOL]

# Everything else is denied, the built-in web tools most of all: they are the exact
# hole this client exists to close.
DISALLOWED_TOOLS = [
    "WebSearch",
    "WebFetch",
    "Agent",
    "Bash",
    "Edit",
    "Glob",
    "Grep",
    "LS",
    "MultiEdit",
    "NotebookEdit",
    "NotebookRead",
    "Read",
    "Skill",
    "Task",
    "ToolSearch",
    "Write",
]

MAX_RESULTS_CAP = 15


def _summarize(result):
    """Compact the search payload; full text would swamp the agent's context."""
    lines = [
        f"Point-in-time search (provider={result['provider']}, cutoff={result['cutoff']})",
        f"query: {result['query']}",
        f"admitted {result['counts']['admitted']} of {result['counts']['returned_by_provider']} "
        f"results; {result['counts']['rejected']} blocked as post-cutoff or undated",
        "",
    ]
    if not result["results"]:
        lines.append("No admissible results. Try a different query or an earlier-dated source.")
    for i, item in enumerate(result["results"], start=1):
        lines.append(f"[{i}] {item['title']}")
        lines.append(f"    url: {item['url']}")
        lines.append(f"    published: {item['published_date']}")
        if item["text"]:
            snippet = " ".join(item["text"].split())[:700]
            lines.append(f"    excerpt: {snippet}")
        lines.append("")
    if result["rejected"]:
        lines.append("Blocked (do not cite, do not reason from these):")
        for item in result["rejected"]:
            lines.append(f"    - {item['url']} — {item['reason']}")
    return "\n".join(lines)


def _build_pit_mcp_server(as_of, provider=None, allow_same_day=False):
    """
    Build an in-process MCP server whose cutoff is fixed at `as_of`. The tools take
    no date argument, so the agent has no way to move the boundary.
    """

    @tool(
        SEARCH_TOOL.rsplit("__", 1)[-1],
        "Search the web as it stood before this run's point-in-time cutoff. Results published "
        "on or after the cutoff are removed before you see them, and undated results are dropped. "
        "This is your ONLY way to search; WebSearch is unavailable.",
        {
            "query": str,
            "num_results": int,
        },
    )
    async def _search(args):
        query_text = (args.get("query") or "").strip()
        if not query_text:
            return {"content": [{"type": "text", "text": "Error: query must not be empty."}]}
        num_results = args.get("num_results") or 10
        try:
            num_results = max(1, min(int(num_results), MAX_RESULTS_CAP))
        except (TypeError, ValueError):
            num_results = 10

        try:
            result = pit_search(
                query_text,
                as_of=as_of,
                num_results=num_results,
                provider=provider,
                allow_same_day=allow_same_day,
            )
        except PitSearchError as e:
            logger.error("PIT search failed for %r: %s", query_text, e)
            return {"content": [{"type": "text", "text": f"Search failed: {e}"}], "is_error": True}

        logger.info(
            "PIT search: %r -> %d admitted / %d blocked",
            query_text, result["counts"]["admitted"], result["counts"]["rejected"],
        )
        return {"content": [{"type": "text", "text": _summarize(result)}]}

    @tool(
        CONTENTS_TOOL.rsplit("__", 1)[-1],
        "Fetch the full text of one URL, subject to the same point-in-time cutoff. A page "
        "published on or after the cutoff is refused. This is your ONLY way to read a page; "
        "WebFetch is unavailable.",
        {"url": str},
    )
    async def _contents(args):
        url = (args.get("url") or "").strip()
        if not url:
            return {"content": [{"type": "text", "text": "Error: url must not be empty."}]}
        try:
            result = pit_get_contents(
                url, as_of=as_of, provider=provider, allow_same_day=allow_same_day
            )
        except PitSearchError as e:
            logger.error("PIT contents failed for %s: %s", url, e)
            return {"content": [{"type": "text", "text": f"Fetch failed: {e}"}], "is_error": True}

        if result["blocked"]:
            text = (
                f"BLOCKED: {url} is not admissible ({result['reason']}). "
                "Do not cite it and do not reason from it."
            )
        else:
            text = (
                f"{url}\npublished: {result['published_date']}\n\n"
                f"{' '.join(result['text'].split())}"
            )
        return {"content": [{"type": "text", "text": text}]}

    return create_sdk_mcp_server(name=MCP_SERVER_NAME, version="1.0.0", tools=[_search, _contents])


async def _deny_non_pit_tools(hook_input, _tool_use_id, _context):
    tool_name = (
        hook_input.get("tool_name", "")
        if isinstance(hook_input, dict)
        else getattr(hook_input, "tool_name", "")
    )
    if tool_name in ALLOWED_TOOLS:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
            },
            "suppressOutput": True,
        }

    reason = (
        f"Tool '{tool_name}' is blocked in a point-in-time run. Only {', '.join(ALLOWED_TOOLS)} "
        "may be used, because they are the only tools that enforce the date cutoff."
    )
    logger.error("Blocking non-PIT tool: %s", tool_name)
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
        "reason": reason,
    }


def _build_agent_options(as_of, stderr_lines, provider=None, allow_same_day=False):
    return ClaudeAgentOptions(
        model=MODEL_ID,
        effort=EFFORT_LEVEL,
        tools=ALLOWED_TOOLS,
        allowed_tools=ALLOWED_TOOLS,
        disallowed_tools=DISALLOWED_TOOLS,
        mcp_servers={MCP_SERVER_NAME: _build_pit_mcp_server(as_of, provider, allow_same_day)},
        hooks={
            "PreToolUse": [
                HookMatcher(matcher=".*", hooks=[_deny_non_pit_tools], timeout=5.0)
            ]
        },
        permission_mode="bypassPermissions",
        cwd=str(_LIVEMACRO_ROOT),
        stderr=_build_stderr_collector(stderr_lines),
    )


PIT_AGENT_INSTRUCTIONS_TEMPLATE = """AGENT INSTRUCTIONS (POINT-IN-TIME RUN):
You are researching ONE data release as if standing at {as_of}. You cannot see past that instant.
1. Your only tools are `{search_tool}` and `{contents_tool}`. WebSearch and WebFetch are disabled and
   any attempt to call them will be refused — do not try.
2. Both tools enforce the cutoff themselves: post-cutoff and undated documents are removed before you
   see them. If a search returns few results, that is the filter working, not a failure. Reformulate
   the query or search for earlier evidence instead of trying to get around it.
3. Search broadly before concluding: regional Fed manufacturing and services surveys, flash PMIs, the
   sub-indices of the previous report, sector news, policy and trade changes, and any disruptions.
4. Use `{contents_tool}` to read the sources that matter rather than relying on excerpts alone.
5. Cite only documents these tools returned to you, with the publication date exactly as reported.
   Never cite a URL you did not retrieve in this run.
6. If you happen to recall the released value from training, that is not admissible evidence. Your
   estimate must follow from the cited pre-cutoff sources.
7. Your FINAL message must be the JSON object alone — no preamble, no explanation, no markdown fences.
"""


async def _run_agent(system_msg, user_msg, as_of, provider=None, allow_same_day=False):
    prompt = (
        f"{system_msg}\n\n"
        f"{user_msg}\n\n"
        + PIT_AGENT_INSTRUCTIONS_TEMPLATE.format(
            as_of=as_of, search_tool=SEARCH_TOOL, contents_tool=CONTENTS_TOOL
        )
    )

    stderr_lines = deque(maxlen=40)
    options = _build_agent_options(as_of, stderr_lines, provider, allow_same_day)
    timeout_secs = _get_agent_timeout_secs()
    state = {"result_text": "", "blocked_attempts": 0, "pit_calls": 0}

    async def _consume():
        stream = query(prompt=prompt, options=options)
        try:
            async for message in stream:
                if hasattr(message, "result"):
                    state["result_text"] = message.result
                elif hasattr(message, "content"):
                    blocks = message.content if isinstance(message.content, list) else [message.content]
                    for block in blocks:
                        if isinstance(block, ToolUseBlock):
                            if block.name in ALLOWED_TOOLS:
                                state["pit_calls"] += 1
                                logger.info("PIT tool call: %s input=%s", block.name, str(block.input)[:150])
                            else:
                                state["blocked_attempts"] += 1
                                logger.error(
                                    "Agent attempted a non-PIT tool: %s input=%s",
                                    block.name, str(block.input)[:150],
                                )
                        elif isinstance(block, ToolResultBlock) and block.is_error:
                            logger.warning("PIT tool error: %s", str(block.content)[:200])
        finally:
            with suppress(Exception):
                await stream.aclose()

    try:
        with _signed_in_sdk_only_env():
            await asyncio.wait_for(_consume(), timeout=timeout_secs)
    except asyncio.TimeoutError as e:
        stderr_summary = " | ".join(stderr_lines)
        raise TimeoutError(
            f"PIT agent timed out after {timeout_secs:.1f}s"
            + (f"\nClaude CLI stderr: {stderr_summary}" if stderr_summary else "")
        ) from e
    except Exception as e:
        if not state["result_text"]:
            stderr_summary = " | ".join(stderr_lines)
            if stderr_summary:
                raise RuntimeError(f"{e}\nClaude CLI stderr: {stderr_summary}") from e
            raise
        logger.warning("PIT agent raised after returning a result; using it. error=%s", e)

    logger.info(
        "PIT agent finished: %d PIT tool calls, %d blocked attempts at other tools",
        state["pit_calls"], state["blocked_attempts"],
    )
    if state["blocked_attempts"]:
        logger.warning(
            "Agent tried %d time(s) to reach an unfiltered tool; all were denied",
            state["blocked_attempts"],
        )
    return state["result_text"]


def generate(system_msg, user_msg, as_of=None, provider=None, allow_same_day=False, **_ignored):
    """
    Interface matches the other clients: generate(system_msg, user_msg) -> (text, citations).

    `as_of` is required — without a cutoff this client has no reason to exist, and
    defaulting it to "now" would quietly turn a backtest into an unfiltered run.
    """
    if not as_of:
        raise ValueError(
            "claude-code-agent-pit requires as_of. Set it on the job, or use claude-code-agent "
            "for an unfiltered live run."
        )

    logger.info("Calling %s with cutoff %s ...", DISPLAY_NAME, as_of)
    result = asyncio.run(_run_agent(system_msg, user_msg, as_of, provider, allow_same_day))
    if not result:
        raise RuntimeError("PIT agent returned empty result")
    logger.info("PIT agent response length: %d chars", len(result))
    return result, None
