"""Shared graph state for the orchestration workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import MessagesState


def merge_dicts(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if a:
        merged.update(a)
    if b:
        merged.update(b)
    return merged


class OrchestratorState(MessagesState, total=False):
    """State flowing through the supervisor graph.

    ``messages`` comes from :class:`MessagesState` (append-only via
    ``add_messages``); the rest is coordination metadata.
    """

    # Task identity
    task: str
    run_id: str
    conversation_id: str

    # Supervisor coordination
    route_history: Annotated[List[str], operator.add]
    last_instructions: str
    agent_outputs: Annotated[Dict[str, str], merge_dicts]
    supervisor_notes: Annotated[List[str], operator.add]
    tool_activity: Annotated[List[dict], operator.add]
    iterations: int
    budget_exhausted: bool

    # Memory
    memories: Annotated[List[str], operator.add]

    # Human-in-the-loop
    escalation_requested: Optional[dict]
    escalation: Optional[dict]
    human_decision: Optional[dict]
    human_feedback_text: str

    # Output
    final_answer: Optional[str]
    status: str


def format_agent_outputs(state: OrchestratorState, limit: int = 2500) -> str:
    outputs = state.get("agent_outputs") or {}
    if not outputs:
        return "(no agent output yet)"
    parts = []
    for name, text in outputs.items():
        clipped = text if len(text) <= limit else text[:limit] + "…"
        parts.append(f"[{name}]\n{clipped}")
    return "\n\n".join(parts)


def format_memories(state: OrchestratorState) -> str:
    memories = state.get("memories") or []
    if not memories:
        return "(none)"
    return "\n".join(f"- {m}" for m in memories[:10])
