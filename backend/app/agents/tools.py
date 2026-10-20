"""Tools available to worker agents.

All tools are constructed per-run via :func:`build_tools` so they close over the
:class:`~app.orchestrator.context.RunContext` (memory + escalation wiring).
Every tool is defensive: failures degrade to informative text instead of
crashing the agent loop.
"""

from __future__ import annotations

import ast
import datetime as _dt
import math
import operator
from typing import TYPE_CHECKING, Any, Callable, List

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool

from app.memory.vector import MemoryHit

if TYPE_CHECKING:  # imported lazily to avoid a package import cycle
    from app.orchestrator.context import RunContext

# ── Safe calculator ───────────────────────────────────────────────────────────

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_SAFE_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
}


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression with a strict AST allow-list."""
    if len(expression) > 500:
        raise ValueError("Expression too long")
    node = ast.parse(expression, mode="eval").body
    return _eval_node(node)


def _eval_node(node: ast.AST, depth: int = 0) -> float:
    if depth > 32:
        raise ValueError("Expression too deeply nested")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and (abs(right) > 1024 or abs(left) > 1e9):
            raise ValueError("Power operation out of allowed range")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, depth + 1))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _SAFE_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"Function not allowed: {node.func.id}")
        return fn(*[_eval_node(a, depth + 1) for a in node.args])
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(a, depth + 1) for a in node.elts)  # type: ignore[return-value]
    if isinstance(node, ast.List):
        return [_eval_node(a, depth + 1) for a in node.elts]  # type: ignore[return-value]
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


# ── Tool factory ──────────────────────────────────────────────────────────────


def build_tools(ctx: RunContext) -> List[BaseTool]:
    """Build the per-run tool belt."""

    @tool
    def web_search(query: str) -> str:
        """Search the web for current information on `query`. Returns top results with titles and snippets."""
        ctx.emit("tool.started", actor="tool:web_search", payload={"query": query})
        results = _duckduckgo_search(query)
        ctx.emit("tool.completed", actor="tool:web_search", payload={"query": query, "hits": len(results)})
        return results

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression, e.g. `(12.5 * 4) / sqrt(2)`. Arithmetic only — no variables or Python statements."""
        try:
            value = safe_eval(expression)
        except Exception as exc:  # noqa: BLE001 - report cleanly to the model
            return f"ERROR: could not evaluate {expression!r}: {exc}"
        ctx.emit("tool.completed", actor="tool:calculator", payload={"expression": expression, "value": value})
        return str(value)

    @tool
    def retrieve_memory(query: str) -> str:
        """Search persistent long-term memory for facts relevant to `query` from previous runs and conversations."""
        hits: list[MemoryHit] = ctx.vector.search(query, k=ctx.settings.memory_top_k)
        ctx.emit("tool.completed", actor="tool:retrieve_memory", payload={"query": query, "hits": len(hits)})
        if not hits:
            return "No relevant memories found."
        return "\n".join(f"- ({h.score:.2f}) {h.text}" for h in hits)

    @tool
    def save_memory(content: str, kind: str = "episodic") -> str:
        """Persist an important fact to long-term memory. `kind` is one of: episodic (event/decision), semantic (general fact), profile (about the user)."""
        kind = kind if kind in {"episodic", "semantic", "profile"} else "episodic"
        record = ctx.store.add_memory_record(
            content=content,
            kind=kind,
            conversation_id=ctx.conversation_id,
            run_id=ctx.run_id,
        )
        ctx.vector.add(
            content,
            metadata={"kind": kind, "run_id": ctx.run_id, "conversation_id": ctx.conversation_id},
            memory_id=record.id,
        )
        ctx.emit("tool.completed", actor="tool:save_memory", payload={"kind": kind, "memory_id": record.id})
        return f"Saved to long-term memory ({kind}, id={record.id})."

    @tool
    def request_human(question: str, context: str = "") -> str:
        """Escalate to a human operator mid-run. Use when you need permission, face an unresolvable ambiguity, or are asked to take a policy-sensitive action. The run pauses until the human responds."""
        ctx.emit("tool.completed", actor="tool:request_human", payload={"question": question})
        return "HUMAN_REQUESTED:" + question  # parsed by the worker into state

    @tool
    def current_datetime() -> str:
        """Return the current UTC date and time."""
        return _dt.datetime.now(_dt.timezone.utc).isoformat()

    return [web_search, calculator, retrieve_memory, save_memory, request_human, current_datetime]


def tools_by_name(tools: List[BaseTool]) -> dict[str, BaseTool]:
    return {t.name: t for t in tools}


# ── Web search (graceful degradation) ────────────────────────────────────────


def _duckduckgo_search(query: str, max_results: int = 4) -> str:
    """Query DuckDuckGo's instant-answer API.

    Degrades gracefully offline: if the network is unavailable the tool says so
    instead of inventing results, and the agent falls back to memory tools.
    """
    try:
        import httpx

        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=5.0,
        )
        data = resp.json()
        lines: list[str] = []
        if data.get("AbstractText"):
            lines.append(f"1. {data.get('Heading', query)} — {data['AbstractText']} (source: {data.get('AbstractSource', 'unknown')})")
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                lines.append(f"- {topic['Text']}")
        if lines:
            return "Search results:\n" + "\n".join(lines[: max_results + 1])
    except Exception:
        pass
    return (
        f"Web search is unavailable in this environment (offline). "
        f"Query was: {query!r}. Use `retrieve_memory` for internal knowledge, "
        f"or reason from context already gathered."
    )


# ── Tool execution helper for agent loops ────────────────────────────────────


def execute_tool_calls(
    tools: List[BaseTool],
    tool_calls: List[dict[str, Any]],
    ctx: RunContext | None = None,
) -> List[ToolMessage]:
    """Run tool calls from an AIMessage and return matching ToolMessages."""
    registry = tools_by_name(tools)
    messages: List[ToolMessage] = []
    for call in tool_calls:
        name = call.get("name", "")
        call_id = call.get("id", "call_0")
        args = call.get("args", {}) or {}
        selected = registry.get(name)
        if selected is None:
            content = f"ERROR: unknown tool '{name}'. Available: {sorted(registry)}"
        else:
            if ctx is not None:
                ctx.emit("tool.started", actor=f"tool:{name}", payload={"args": args})
            try:
                content = str(selected.invoke(args))
            except Exception as exc:  # noqa: BLE001
                content = f"ERROR: tool '{name}' failed: {exc}"
            if ctx is not None:
                ctx.emit("tool.finished", actor=f"tool:{name}", payload={"args": args, "ok": not content.startswith("ERROR")})
        messages.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
    return messages
