"""The supervisor: a LangGraph router that delegates, escalates and finalises.

Graph topology::

    START -> retrieve_memories -> supervisor ─┬─> researcher ─┐
                                             ├─> analyst    ─┤
                                             ├─> writer     ─┼─> supervisor
                                             ├─> escalate   ─┘   (loop)
                                             └─> finalize -> [review] -> END

* ``escalate`` and ``review`` are LangGraph interrupts: the run persists, a
  durable :class:`Escalation` record is created, and a human decision resumes
  the graph via ``Command(resume=...)``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.workers import default_specs, make_worker_node
from app.config import Settings
from app.llm.prompts import (
    SUPERVISOR_CONTEXT,
    SUPERVISOR_SYSTEM,
    WORKER_ROSTER,
    WORKER_MISSIONS,
)
from app.orchestrator.context import RunContext
from app.orchestrator.escalation import EscalationManager, PolicyEngine
from app.orchestrator.state import (
    OrchestratorState,
    format_agent_outputs,
    format_memories,
)

# Route keys used by the conditional edges
ROUTE_WORKERS = ("researcher", "analyst", "writer")
VALID_ACTIONS = set(ROUTE_WORKERS) | {"escalate", "FINISH"}


# ── JSON parsing helpers ─────────────────────────────────────────────────────

def parse_json_block(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output (tolerates fences/prose)."""
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return {}


def parse_supervisor_action(text: str) -> dict[str, Any]:
    """Parse + validate a supervisor decision, with safe defaults."""
    raw = parse_json_block(text)
    action = str(raw.get("action", "")).strip()
    if action not in VALID_ACTIONS:
        lowered = text.lower()
        for candidate in ("researcher", "analyst", "writer", "escalate", "finish"):
            if f'"{candidate}"' in lowered or f" {candidate}" in lowered:
                action = "FINISH" if candidate == "finish" else candidate
                break
        else:
            action = "FINISH"
    try:
        confidence = float(raw.get("confidence", 0.7))
    except (TypeError, ValueError):
        confidence = 0.7
    confidence = min(1.0, max(0.0, confidence))
    return {
        "action": action,
        "reasoning": str(raw.get("reasoning", "")),
        "instructions": str(raw.get("instructions", "")),
        "confidence": confidence,
        "escalation_question": str(raw.get("escalation_question", "")),
    }


# ── Graph node factories ─────────────────────────────────────────────────────

def make_supervisor_node(ctx: RunContext, settings: Settings) -> Callable[[OrchestratorState], dict]:
    policy = PolicyEngine(settings)

    def node(state: OrchestratorState) -> dict[str, Any]:
        from app.llm.providers import get_chat_model

        iterations = int(state.get("iterations") or 0)
        roster = WORKER_ROSTER.format(**WORKER_MISSIONS)
        system = SUPERVISOR_SYSTEM.format(worker_roster=roster)

        budget_note = (
            "BUDGET EXHAUSTED: you must route to FINISH now, accepting the best available answer."
            if state.get("budget_exhausted")
            else ""
        )
        context_prompt = SUPERVISOR_CONTEXT.format(
            task=state.get("task", ""),
            route_history=" -> ".join(state.get("route_history") or []) or "(none)",
            agent_outputs=format_agent_outputs(state),
            memories=format_memories(state),
            human_feedback=state.get("human_feedback_text") or "None yet.",
            iterations=iterations,
            max_iterations=settings.max_supervisor_iterations,
            budget_note=budget_note,
        )

        llm = get_chat_model("supervisor", settings)
        response = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=context_prompt)]
        )
        decision = parse_supervisor_action(str(response.content or ""))
        action = decision["action"]

        ctx.emit(
            "supervisor.route",
            actor="supervisor",
            payload={
                "action": action,
                "confidence": decision["confidence"],
                "reasoning": decision["reasoning"][:300],
            },
        )

        # ── Escalation policy gate (applies to every route) ─────────────────
        tool_request = state.get("escalation_requested")
        check = policy.evaluate(
            task=state.get("task", ""),
            plan=decision.get("instructions", ""),
            confidence=decision["confidence"],
            explicit=(action == "escalate"),
            explicit_question=decision.get("escalation_question", ""),
            tool_request=tool_request,
            iterations=iterations,
            already_reviewed=bool(state.get("human_decision")),
        )

        route = action
        escalation_payload: dict[str, Any] | None = None
        # FINISH normally outranks the policy gate so runs can always terminate —
        # but an unanswered agent `request_human` must never be silently dropped.
        gate_open = action != "FINISH" or check.source == "agent_request"
        if check.needs_escalation and gate_open:
            route = "escalate"
            escalation_payload = {
                "type": "decision",
                "reason": check.reason,
                "question": check.question,
                "source": check.source,
                "matched_rules": check.matched_rules,
                "context": {
                    "task": state.get("task", ""),
                    "planned_action": action,
                    "planned_instructions": decision.get("instructions", ""),
                    "confidence": decision["confidence"],
                    "route_history": state.get("route_history", []),
                    "agent_outputs": {
                        k: v[:500] for k, v in (state.get("agent_outputs") or {}).items()
                    },
                },
            }
            ctx.emit("supervisor.escalate", actor="supervisor", payload=escalation_payload)

        update: dict[str, Any] = {
            "route_history": [route],
            "last_instructions": decision.get("instructions", ""),
            "supervisor_notes": [
                f"[turn {iterations + 1}] {action} (confidence {decision['confidence']:.2f}): {decision['reasoning'][:240]}"
            ],
            "iterations": iterations + 1,
            "status": "awaiting_human" if route == "escalate" else "running",
        }
        if route == "FINISH" and not state.get("budget_exhausted") and iterations + 1 >= settings.max_supervisor_iterations:
            update["budget_exhausted"] = True
        if escalation_payload:
            update["escalation"] = escalation_payload
            update["budget_exhausted"] = False
        return update

    node.__name__ = "supervisor"
    return node


def make_memory_node(ctx: RunContext) -> Callable[[OrchestratorState], dict]:
    """Retrieve long-term memories relevant to the task (run entry point)."""

    def node(state: OrchestratorState) -> dict[str, Any]:
        task = state.get("task", "")
        hits = ctx.vector.search(task, k=ctx.settings.memory_top_k)
        memories = [h.text for h in hits]
        if memories:
            ctx.emit("memory.retrieved", actor="memory", payload={"count": len(memories)})
        return {"memories": memories, "status": "running"}

    node.__name__ = "retrieve_memories"
    return node


def make_escalate_node(ctx: RunContext) -> Callable[[OrchestratorState], dict]:
    """Pause the run for a human decision (LangGraph interrupt)."""

    def node(state: OrchestratorState) -> dict[str, Any]:
        from langgraph.types import interrupt

        payload = state.get("escalation") or {
            "type": "decision",
            "reason": "Supervisor requested human judgement.",
            "question": "How should the team proceed?",
            "source": "supervisor",
            "context": {"task": state.get("task", "")},
        }
        # Blocks here until a human resolves the escalation via the API/UI.
        decision = interrupt(payload)
        if not isinstance(decision, dict):
            decision = {"decision": str(decision)}
        update = EscalationManager.apply_decision_to_state(decision)
        update["supervisor_notes"] = [
            f"Human decision received: {decision.get('decision', 'approve')} — {decision.get('comment', '')}".strip(" —")
        ]
        ctx.emit("human.decision", actor="human", payload=decision)
        return update

    node.__name__ = "escalate"
    return node


def make_finalize_node(ctx: RunContext, settings: Settings) -> Callable[[OrchestratorState], dict]:
    """Compose the final answer from worker outputs."""

    def node(state: OrchestratorState) -> dict[str, Any]:
        outputs = state.get("agent_outputs") or {}
        final = outputs.get("writer") or outputs.get("analyst") or outputs.get("researcher")
        if not final:
            final = (
                "The team did not produce a complete answer. Partial findings:\n\n"
                + format_agent_outputs(state)
            )
        decision = state.get("human_decision") or {}
        if decision.get("decision") == "modify" and decision.get("modified_answer"):
            final = str(decision["modified_answer"])
        ctx.emit("run.finalized", actor="orchestrator", payload={"length": len(final)})
        return {
            "final_answer": final,
            "status": "awaiting_review" if settings.require_final_approval else "completed",
        }

    node.__name__ = "finalize"
    return node


def make_review_node(ctx: RunContext, settings: Settings) -> Callable[[OrchestratorState], dict]:
    """Optional human approval gate on the final answer (interrupt)."""

    def node(state: OrchestratorState) -> dict[str, Any]:
        from langgraph.types import interrupt

        draft = state.get("final_answer") or ""
        decision = interrupt(
            {
                "type": "final_review",
                "reason": "Final answer awaiting human approval.",
                "question": "Approve this answer before it is shown to the user?",
                "context": {"task": state.get("task", ""), "draft": draft},
            }
        )
        if not isinstance(decision, dict):
            decision = {"decision": str(decision)}
        verdict = str(decision.get("decision", "approve")).lower()
        ctx.emit("human.review", actor="human", payload={"decision": verdict})
        if verdict == "approve":
            return {
                "status": "completed",
                "final_answer": draft,
                "human_feedback_text": "Human approved the final answer.",
            }
        # Rejected / modified — send back through the supervisor with feedback.
        update = EscalationManager.apply_decision_to_state(decision)
        update["status"] = "running"
        if verdict == "modify" and decision.get("modified_answer"):
            update["final_answer"] = str(decision["modified_answer"])
            update["status"] = "completed"
            return update
        return update

    node.__name__ = "review"
    return node


def route_after_supervisor(state: OrchestratorState) -> str:
    history = state.get("route_history") or []
    return history[-1] if history else "FINISH"


def route_after_finalize(state: OrchestratorState) -> str:
    return "review" if state.get("status") == "awaiting_review" else "end"


def route_after_review(state: OrchestratorState) -> str:
    return "end" if state.get("status") == "completed" else "supervisor"


# ── Graph assembly ───────────────────────────────────────────────────────────

def build_graph(ctx: RunContext, settings: Settings, checkpointer: Any = None):
    """Compile the supervisor orchestration graph."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(OrchestratorState)

    graph.add_node("retrieve_memories", make_memory_node(ctx))
    graph.add_node("supervisor", make_supervisor_node(ctx, settings))
    for name, spec in default_specs().items():
        graph.add_node(name, make_worker_node(spec, ctx, settings))
    graph.add_node("escalate", make_escalate_node(ctx))
    graph.add_node("finalize", make_finalize_node(ctx, settings))
    graph.add_node("review", make_review_node(ctx, settings))

    graph.add_edge(START, "retrieve_memories")
    graph.add_edge("retrieve_memories", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "escalate": "escalate",
            "FINISH": "finalize",
        },
    )
    for worker in ROUTE_WORKERS:
        graph.add_edge(worker, "supervisor")
    graph.add_edge("escalate", "supervisor")
    graph.add_conditional_edges(
        "finalize",
        route_after_finalize,
        {"review": "review", "end": END},
    )
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {"supervisor": "supervisor", "end": END},
    )

    return graph.compile(checkpointer=checkpointer)
