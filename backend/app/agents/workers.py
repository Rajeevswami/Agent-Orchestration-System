"""Worker agents: bounded tool-use loops under supervisor direction.

Each worker is a LangGraph node created by :func:`make_worker_node`. Inside the
node the worker runs a short ReAct-style loop (model -> tools -> model ...) until
it produces a final text answer or hits its step budget, then reports back to
the supervisor state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from app.agents.tools import build_tools, execute_tool_calls
from app.config import Settings
from app.llm.prompts import WORKER_CONTEXT, WORKER_MISSIONS, WORKER_SYSTEM
from app.orchestrator.context import RunContext
from app.orchestrator.state import OrchestratorState, format_agent_outputs, format_memories


@dataclass
class WorkerSpec:
    name: str            # routing key used by the supervisor
    role: str            # human label
    mission: str


def default_specs() -> dict[str, WorkerSpec]:
    return {
        "researcher": WorkerSpec("researcher", "Researcher", WORKER_MISSIONS["researcher"]),
        "analyst": WorkerSpec("analyst", "Analyst", WORKER_MISSIONS["analyst"]),
        "writer": WorkerSpec("writer", "Writer", WORKER_MISSIONS["writer"]),
    }


HUMAN_REQUEST_PREFIX = "HUMAN_REQUESTED:"


def make_worker_node(
    spec: WorkerSpec, ctx: RunContext, settings: Settings
) -> Callable[[OrchestratorState], dict[str, Any]]:
    """Return a graph node running one worker agent."""

    tools = build_tools(ctx)
    tool_names = ", ".join(t.name for t in tools)

    def node(state: OrchestratorState) -> dict[str, Any]:
        from app.llm.providers import get_chat_model  # local import avoids cycles

        instructions = state.get("last_instructions") or state.get("task", "")
        system_prompt = (
            WORKER_SYSTEM.format(role=spec.role, mission=spec.mission, tools=tool_names)
            + "\n\n"
            + WORKER_CONTEXT.format(
                instructions=instructions,
                task=state.get("task", ""),
                agent_outputs=format_agent_outputs(state),
                memories=format_memories(state),
                human_feedback=state.get("human_feedback_text") or "None yet.",
            )
        )

        llm = get_chat_model(spec.name, settings)
        if hasattr(llm, "bind_tools"):
            try:
                llm = llm.bind_tools(tools)
            except Exception:  # pragma: no cover - provider without tool binding
                pass

        conversation: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        ctx.emit("worker.started", actor=spec.name, payload={"instructions": instructions})

        final_text = ""
        requested_human: str | None = None
        for step in range(settings.max_tool_steps):
            ai: AIMessage = llm.invoke(conversation)
            conversation.append(ai)
            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                final_text = str(ai.content or "")
                break

            # Notice human-escalation requests before executing the tool.
            for call in tool_calls:
                if call.get("name") == "request_human":
                    question = (call.get("args") or {}).get("question", "Human input required.")
                    requested_human = question

            tool_messages = execute_tool_calls(tools, tool_calls, ctx)
            conversation.extend(tool_messages)

            for tm in tool_messages:
                if tm.content.startswith(HUMAN_REQUEST_PREFIX):
                    requested_human = tm.content[len(HUMAN_REQUEST_PREFIX):]
        else:
            # Budget exhausted — summarise whatever the model last produced.
            last = [m for m in reversed(conversation) if isinstance(m, AIMessage)]
            final_text = str(last[0].content if last else "Worker hit its step budget.")

        summary = final_text.strip() or f"{spec.role} produced no text output."
        ctx.emit("worker.completed", actor=spec.name, payload={"summary": summary[:400]})

        update: dict[str, Any] = {
            "messages": [AIMessage(content=summary, name=spec.name)],
            "agent_outputs": {spec.name: summary},
            "tool_activity": [
                {"agent": spec.name, "turn": i, "content": str(m.content)[:200]}
                for i, m in enumerate(conversation)
                if not isinstance(m, SystemMessage)
            ],
            "last_instructions": "",
        }
        if requested_human:
            update["escalation_requested"] = {
                "source": spec.name,
                "question": requested_human,
            }
        return update

    node.__name__ = f"node_{spec.name}"
    return node
