"""Relational store: CRUD helpers over the episodic memory tables."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import desc, select

from app.memory.db import Database
from app.memory.models import (
    Conversation,
    Escalation,
    MemoryRecord,
    Message,
    Run,
    RunEvent,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RelationalStore:
    """All persistent state that must survive restarts goes through here."""

    def __init__(self, db: Database):
        self.db = db

    # ── Conversations ────────────────────────────────────────────────────────
    def create_conversation(self, title: str = "New conversation", **meta: Any) -> Conversation:
        with self.db.session() as s:
            conv = Conversation(title=title, metadata_=meta or {})
            s.add(conv)
            s.flush()
            return conv

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self.db.session() as s:
            return s.get(Conversation, conversation_id)

    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        with self.db.session() as s:
            stmt = select(Conversation).order_by(desc(Conversation.created_at)).limit(limit)
            return list(s.scalars(stmt))

    # ── Runs ─────────────────────────────────────────────────────────────────
    def create_run(
        self,
        conversation_id: str,
        task: str,
        provider: str = "",
        **input_data: Any,
    ) -> Run:
        with self.db.session() as s:
            run = Run(
                conversation_id=conversation_id,
                task=task,
                provider=provider,
                status="queued",
                input_data=input_data or {},
            )
            s.add(run)
            s.flush()
            return run

    def get_run(self, run_id: str) -> Run | None:
        with self.db.session() as s:
            return s.get(Run, run_id)

    def list_runs(self, limit: int = 50, conversation_id: str | None = None) -> list[Run]:
        with self.db.session() as s:
            stmt = select(Run).order_by(desc(Run.created_at)).limit(limit)
            if conversation_id:
                stmt = stmt.where(Run.conversation_id == conversation_id)
            return list(s.scalars(stmt))

    def update_run(self, run_id: str, **fields: Any) -> Run | None:
        with self.db.session() as s:
            run = s.get(Run, run_id)
            if run is None:
                return None
            for key, value in fields.items():
                if key == "output_data" and isinstance(value, dict):
                    merged = dict(run.output_data or {})
                    merged.update(value)
                    value = merged
                setattr(run, key, value)
            s.flush()
            return run

    # ── Messages ─────────────────────────────────────────────────────────────
    def add_message(
        self,
        run_id: str,
        conversation_id: str,
        role: str,
        content: str,
        name: str | None = None,
        tool_calls: list | None = None,
        seq: int = 0,
    ) -> Message:
        with self.db.session() as s:
            msg = Message(
                run_id=run_id,
                conversation_id=conversation_id,
                role=role,
                name=name,
                content=content or "",
                tool_calls=tool_calls,
                seq=seq,
            )
            s.add(msg)
            s.flush()
            return msg

    def add_messages(self, rows: Iterable[dict[str, Any]]) -> int:
        count = 0
        for row in rows:
            self.add_message(**row)
            count += 1
        return count

    def replace_messages(self, run_id: str, rows: Iterable[dict[str, Any]]) -> int:
        """Rewrite a run's message history (graph checkpoints are authoritative)."""
        from sqlalchemy import delete

        with self.db.session() as s:
            s.execute(delete(Message).where(Message.run_id == run_id))
        return self.add_messages(rows)

    def get_messages(self, conversation_id: str | None = None, run_id: str | None = None) -> list[Message]:
        with self.db.session() as s:
            stmt = select(Message).order_by(Message.seq, Message.id)
            if conversation_id:
                stmt = stmt.where(Message.conversation_id == conversation_id)
            if run_id:
                stmt = stmt.where(Message.run_id == run_id)
            return list(s.scalars(stmt))

    # ── Events ───────────────────────────────────────────────────────────────
    def add_event(self, run_id: str, type: str, actor: str = "", payload: dict | None = None) -> RunEvent:
        with self.db.session() as s:
            ev = RunEvent(run_id=run_id, type=type, actor=actor, payload=payload or {})
            s.add(ev)
            s.flush()
            return ev

    def list_events(self, run_id: str, since_id: int = 0) -> list[RunEvent]:
        with self.db.session() as s:
            stmt = (
                select(RunEvent)
                .where(RunEvent.run_id == run_id, RunEvent.id > since_id)
                .order_by(RunEvent.id)
            )
            return list(s.scalars(stmt))

    # ── Escalations ──────────────────────────────────────────────────────────
    def create_escalation(
        self,
        run_id: str,
        kind: str,
        reason: str,
        question: str,
        context: dict | None = None,
    ) -> Escalation:
        # Idempotent: one pending escalation per run at a time.
        existing = self.get_pending_escalation(run_id)
        if existing is not None:
            return existing
        with self.db.session() as s:
            esc = Escalation(
                run_id=run_id,
                kind=kind,
                reason=reason,
                question=question,
                context=context or {},
                status="pending",
            )
            s.add(esc)
            s.flush()
            return esc

    def get_escalation(self, escalation_id: str) -> Escalation | None:
        with self.db.session() as s:
            return s.get(Escalation, escalation_id)

    def get_pending_escalation(self, run_id: str) -> Escalation | None:
        with self.db.session() as s:
            stmt = (
                select(Escalation)
                .where(Escalation.run_id == run_id, Escalation.status == "pending")
                .order_by(desc(Escalation.created_at))
                .limit(1)
            )
            return s.scalars(stmt).first()

    def list_escalations(self, status: str | None = None, limit: int = 50) -> list[Escalation]:
        with self.db.session() as s:
            stmt = select(Escalation).order_by(desc(Escalation.created_at)).limit(limit)
            if status:
                stmt = stmt.where(Escalation.status == status)
            return list(s.scalars(stmt))

    def resolve_escalation(
        self, escalation_id: str, status: str, decision: dict
    ) -> Escalation | None:
        with self.db.session() as s:
            esc = s.get(Escalation, escalation_id)
            if esc is None:
                return None
            esc.status = status
            esc.decision = decision
            esc.resolved_at = _utcnow()
            s.flush()
            return esc

    # ── Memory records ───────────────────────────────────────────────────────
    def add_memory_record(
        self,
        content: str,
        kind: str = "episodic",
        conversation_id: str | None = None,
        run_id: str | None = None,
        **metadata: Any,
    ) -> MemoryRecord:
        with self.db.session() as s:
            rec = MemoryRecord(
                content=content,
                kind=kind,
                conversation_id=conversation_id,
                run_id=run_id,
                metadata_=metadata or {},
            )
            s.add(rec)
            s.flush()
            return rec

    def list_memory_records(self, kind: str | None = None, limit: int = 100) -> list[MemoryRecord]:
        with self.db.session() as s:
            stmt = select(MemoryRecord).order_by(desc(MemoryRecord.created_at)).limit(limit)
            if kind:
                stmt = stmt.where(MemoryRecord.kind == kind)
            return list(s.scalars(stmt))
