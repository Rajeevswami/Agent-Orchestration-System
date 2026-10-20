"""Tool-belt unit tests: safe calculator, memory tools, escalation tool."""

from __future__ import annotations

import pytest

from app.agents.tools import build_tools, execute_tool_calls, safe_eval
from app.orchestrator.context import RunContext


def test_safe_eval_arithmetic():
    assert safe_eval("1 + 2 * 3") == 7
    assert safe_eval("(12.5 * 4) / 2") == 25.0
    assert safe_eval("sqrt(16)") == 4.0
    assert safe_eval("2 ** 10") == 1024
    assert safe_eval("min(3, 9, 4)") == 3


def test_safe_eval_rejects_unsafe_expressions():
    with pytest.raises(Exception):
        safe_eval("__import__('os').system('ls')")
    with pytest.raises(Exception):
        safe_eval("open('/etc/passwd')")
    with pytest.raises(Exception):
        safe_eval("[x for x in range(10)]")


def _ctx(service) -> RunContext:
    return RunContext(
        run_id="r1",
        conversation_id="c1",
        settings=service.settings,
        store=service.store,
        vector=service.vector,
    )


def test_memory_tools_roundtrip(service):
    ctx = _ctx(service)
    tools = build_tools(ctx)
    result = execute_tool_calls(
        tools,
        [{"name": "save_memory", "args": {"content": "The launch date is March 12", "kind": "semantic"}, "id": "t1"}],
        ctx,
    )
    assert "Saved" in result[0].content

    hits = execute_tool_calls(
        tools, [{"name": "retrieve_memory", "args": {"query": "launch date"}, "id": "t2"}], ctx
    )
    assert "March 12" in hits[0].content


def test_calculator_tool_reports_errors_cleanly(service):
    ctx = _ctx(service)
    tools = build_tools(ctx)
    (msg,) = execute_tool_calls(
        tools, [{"name": "calculator", "args": {"expression": "1 +"}, "id": "t1"}], ctx
    )
    assert msg.content.startswith("ERROR")


def test_request_human_tool_flags_escalation(service):
    ctx = _ctx(service)
    tools = build_tools(ctx)
    (msg,) = execute_tool_calls(
        tools,
        [{"name": "request_human", "args": {"question": "May I send the email?"}, "id": "t1"}],
        ctx,
    )
    assert msg.content.startswith("HUMAN_REQUESTED:")
    assert "May I send the email?" in msg.content


def test_unknown_tool_returns_error_message(service):
    ctx = _ctx(service)
    tools = build_tools(ctx)
    (msg,) = execute_tool_calls(tools, [{"name": "nope", "args": {}, "id": "t1"}], ctx)
    assert "unknown tool" in msg.content
