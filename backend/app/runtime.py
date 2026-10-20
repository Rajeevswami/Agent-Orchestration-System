"""RunService: the operational core that executes, persists and resumes runs.

Responsibilities:

* create conversation/run records and dispatch execution (inline or Celery);
* build a per-run :class:`RunContext` + compiled LangGraph with a durable
  checkpointer (Postgres/SQLite) so interrupted runs survive restarts;
* translate LangGraph interrupts into durable :class:`Escalation` records;
* resume runs with human decisions via ``Command(resume=...)``;
* persist messages and consolidate long-term memory after completion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app.config import Settings, get_settings
from app.memory.db import Database
from app.memory.models import Run
from app.memory.store import RelationalStore
from app.memory.vector import VectorMemory
from app.orchestrator.context import RunContext
from app.orchestrator.escalation import EscalationManager
from app.orchestrator.supervisor import build_graph, parse_json_block

logger = logging.getLogger("agent_orch.runtime")


class RunService:
    """Creates, executes and resumes orchestration runs."""

    def __init__(
        self,
        settings: Settings | None = None,
        database: Database | None = None,
        store: RelationalStore | None = None,
        vector: VectorMemory | None = None,
        checkpointer: Any = None,
    ):
        self.settings = settings or get_settings()
        self.database = database or Database(self.settings.database_url)
        self.database.init_db()
        self.store = store or RelationalStore(self.database)
        self.vector = vector or VectorMemory(self.settings)
        self.escalations = EscalationManager(self.store, self.settings)
        self._checkpointer = checkpointer if checkpointer is not None else self._make_checkpointer()

    # ── Checkpointer ─────────────────────────────────────────────────────────
    def _make_checkpointer(self) -> Any:
        """Durable LangGraph checkpointer matching the configured database."""
        url = self.settings.database_url
        try:
            if url.startswith("postgresql"):
                import psycopg
                from langgraph.checkpoint.postgres import PostgresSaver

                conn = psycopg.connect(url, autocommit=True)
                saver = PostgresSaver(conn)
                saver.setup()
                return saver
            if url.startswith("sqlite"):
                path = url.replace("sqlite:///", "", 1) or "checkpoints.sqlite"
                if path == ":memory:":
                    from langgraph.checkpoint.memory import InMemorySaver

                    return InMemorySaver()
                import sqlite3

                from langgraph.checkpoint.sqlite import SqliteSaver

                Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(path, check_same_thread=False)
                return SqliteSaver(conn)
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.warning("Falling back to in-memory checkpointer: %s", exc)
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    # ── Run lifecycle ────────────────────────────────────────────────────────
    def create_conversation(self, title: str) -> Any:
        return self.store.create_conversation(title=title)

    def create_run(self, task: str, conversation_id: str | None = None, **meta: Any) -> Run:
        if conversation_id is None:
            conversation = self.store.create_conversation(title=task[:60])
            conversation_id = conversation.id
        return self.store.create_run(
            conversation_id=conversation_id,
            task=task,
            provider=self.settings.llm_provider,
            **meta,
        )

    def submit_run(self, task: str, conversation_id: str | None = None, mode: str = "async", **meta: Any) -> Run:
        """Create a run and dispatch it (Celery or inline)."""
        run = self.create_run(task, conversation_id=conversation_id, **meta)
        if mode == "sync":
            self.execute_run(run.id)
            return self.store.get_run(run.id) or run
        if self.settings.queue_mode == "celery":
            from app.queue.tasks import execute_run_task

            execute_run_task.delay(run.id)
        else:
            self.execute_run(run.id)
        return self.store.get_run(run.id) or run

    def execute_run(self, run_id: str) -> dict[str, Any]:
        """Execute (or continue) a run to completion / its next interrupt."""
        run = self.store.get_run(run_id)
        if run is None:
            raise ValueError(f"Unknown run: {run_id}")
        self.store.update_run(run_id, status="running", started_at=_now())

        # Per-run policy overrides from the request metadata (e.g. UI toggle).
        settings = self.settings
        overrides = {
            k: v
            for k, v in (run.input_data or {}).items()
            if k in ("require_final_approval", "max_supervisor_iterations", "min_confidence")
        }
        if overrides:
            settings = settings.model_copy(update=overrides)

        ctx = RunContext(
            run_id=run.id,
            conversation_id=run.conversation_id,
            settings=settings,
            store=self.store,
            vector=self.vector,
        )
        app = build_graph(ctx, settings, checkpointer=self._checkpointer)
        config = {"configurable": {"thread_id": run.id}}
        initial = {
            "task": run.task,
            "run_id": run.id,
            "conversation_id": run.conversation_id,
            "messages": [HumanMessage(content=run.task)],
            "route_history": [],
            "agent_outputs": {},
            "supervisor_notes": [],
            "tool_activity": [],
            "memories": [],
            "iterations": 0,
            "budget_exhausted": False,
            "status": "running",
        }

        try:
            result = app.invoke(initial, config)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Run %s failed", run_id)
            self.store.update_run(run_id, status="failed", error=str(exc), finished_at=_now())
            ctx.emit("run.failed", actor="orchestrator", payload={"error": str(exc)})
            return {"status": "failed", "error": str(exc)}

        return self._settle(ctx, run, result)

    def resume_run(self, run_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        """Feed a human decision back into an interrupted run."""
        run = self.store.get_run(run_id)
        if run is None:
            raise ValueError(f"Unknown run: {run_id}")
        pending = self.store.get_pending_escalation(run_id)
        if pending is not None:
            verdict = str(decision.get("decision", "approve")).lower()
            status = {"approve": "approved", "reject": "rejected", "modify": "modified"}.get(verdict, "approved")
            self.store.resolve_escalation(pending.id, status=status, decision=decision)

        settings = self.settings
        overrides = {
            k: v
            for k, v in (run.input_data or {}).items()
            if k in ("require_final_approval", "max_supervisor_iterations", "min_confidence")
        }
        if overrides:
            settings = settings.model_copy(update=overrides)

        ctx = RunContext(
            run_id=run.id,
            conversation_id=run.conversation_id,
            settings=settings,
            store=self.store,
            vector=self.vector,
        )
        app = build_graph(ctx, settings, checkpointer=self._checkpointer)
        config = {"configurable": {"thread_id": run.id}}

        try:
            result = app.invoke(Command(resume=decision), config)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Resume of run %s failed", run_id)
            self.store.update_run(run_id, status="failed", error=str(exc), finished_at=_now())
            return {"status": "failed", "error": str(exc)}

        self.store.update_run(run_id, status="running")
        return self._settle(ctx, run, result)

    # ── Post-execution settlement ────────────────────────────────────────────
    def _settle(self, ctx: RunContext, run: Run, result: dict[str, Any]) -> dict[str, Any]:
        interrupts = result.get("__interrupt__") or []
        if interrupts:
            payload = getattr(interrupts[0], "value", interrupts[0])
            kind = payload.get("type", "decision") if isinstance(payload, dict) else "decision"
            reason = payload.get("reason", "") if isinstance(payload, dict) else ""
            question = payload.get("question", "") if isinstance(payload, dict) else str(payload)
            context = payload.get("context", {}) if isinstance(payload, dict) else {}
            escalation = self.escalations.create(
                run_id=run.id,
                kind=kind if kind in ("decision", "final_review") else "decision",
                reason=reason,
                question=question,
                context=context,
            )
            self.store.update_run(run.id, status="awaiting_human")
            self._persist_messages(run, result)
            payload_dict = payload if isinstance(payload, dict) else {}
            return {
                "status": "awaiting_human",
                "run_id": run.id,
                "escalation_id": escalation.id,
                "escalation": {
                    "id": escalation.id,
                    "kind": escalation.kind,
                    "type": payload_dict.get("type", escalation.kind),
                    "reason": escalation.reason,
                    "question": escalation.question,
                    "source": payload_dict.get("source", ""),
                    "context": escalation.context,
                },
            }

        final_answer = result.get("final_answer")
        status = result.get("status") or "completed"
        if status == "awaiting_review":
            status = "awaiting_human"
        self.store.update_run(
            run.id,
            status=status if status in ("completed", "failed") else "completed",
            final_answer=final_answer,
            finished_at=_now(),
            output_data={
                "route_history": result.get("route_history", []),
                "supervisor_notes": result.get("supervisor_notes", []),
                "agent_outputs": result.get("agent_outputs", {}),
            },
        )
        self._persist_messages(run, result)
        self._consolidate_memory(run, result)
        return {
            "status": "completed",
            "run_id": run.id,
            "final_answer": final_answer,
        }

    def _persist_messages(self, run: Run, result: dict[str, Any]) -> None:
        """Sync graph messages into the relational history (checkpoint is source of truth)."""
        rows = []
        seq = 0
        for msg in result.get("messages") or []:
            role, name, tool_calls = _message_meta(msg)
            rows.append(
                dict(
                    run_id=run.id,
                    conversation_id=run.conversation_id,
                    role=role,
                    name=name,
                    content=str(getattr(msg, "content", "") or ""),
                    tool_calls=tool_calls,
                    seq=seq,
                )
            )
            seq += 1
        if rows:
            self.store.replace_messages(run.id, rows)

    def _consolidate_memory(self, run: Run, result: dict[str, Any]) -> None:
        """Persist run outcomes into long-term memory (both stores)."""
        if not self.settings.memory_autosave:
            return
        try:
            final = (result.get("final_answer") or "")[:600]
            facts = [f"Task: {run.task[:200]}"]
            if final:
                facts.append(f"Outcome: {final}")
            for note in (result.get("supervisor_notes") or [])[-3:]:
                facts.append(f"Supervisor: {note[:200]}")
            content = "\n".join(facts)
            record = self.store.add_memory_record(
                content=content,
                kind="episodic",
                conversation_id=run.conversation_id,
                run_id=run.id,
            )
            self.vector.add(
                content,
                metadata={"kind": "episodic", "run_id": run.id, "conversation_id": run.conversation_id},
                memory_id=record.id,
            )
        except Exception:  # pragma: no cover - memory must never fail a run
            logger.exception("Memory consolidation failed for run %s", run.id)

    # ── Introspection ────────────────────────────────────────────────────────
    def pending_escalation(self, run_id: str) -> Any:
        return self.store.get_pending_escalation(run_id)


def _message_meta(msg: BaseMessage) -> tuple[str, str | None, list | None]:
    if isinstance(msg, HumanMessage):
        return "user", None, None
    if isinstance(msg, ToolMessage):
        return "tool", getattr(msg, "name", None), None
    if isinstance(msg, AIMessage):
        name = getattr(msg, "name", None)
        calls = getattr(msg, "tool_calls", None) or None
        return ("agent" if name else "assistant"), name, calls
    return "system", getattr(msg, "name", None), None


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
