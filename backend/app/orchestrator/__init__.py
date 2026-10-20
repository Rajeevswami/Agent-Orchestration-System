"""LangGraph orchestration: supervisor graph, workers, escalation interrupts.

Exports are resolved lazily (PEP 562) so importing submodules in any order —
``app.agents`` <-> ``app.orchestrator`` included — never hits an import cycle.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "RunContext",
    "OrchestratorState",
    "PolicyEngine",
    "EscalationManager",
    "EscalationCheck",
    "build_graph",
    "parse_supervisor_action",
]

_EXPORTS = {
    "RunContext": ("app.orchestrator.context", "RunContext"),
    "OrchestratorState": ("app.orchestrator.state", "OrchestratorState"),
    "PolicyEngine": ("app.orchestrator.escalation", "PolicyEngine"),
    "EscalationManager": ("app.orchestrator.escalation", "EscalationManager"),
    "EscalationCheck": ("app.orchestrator.escalation", "EscalationCheck"),
    "build_graph": ("app.orchestrator.supervisor", "build_graph"),
    "parse_supervisor_action": ("app.orchestrator.supervisor", "parse_supervisor_action"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        import importlib

        module_name, attr = _EXPORTS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
