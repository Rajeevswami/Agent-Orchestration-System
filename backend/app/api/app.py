"""FastAPI application: REST API + live web console.

The API is the single integration surface for the Streamlit frontend, the
built-in web console (``/console``) and curl/CLI clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text as sql_text

from app import __version__
from app.api.schemas import (
    ConversationCreate,
    EscalationDecision,
    MemorySearch,
    RunCreate,
)
from app.config import get_settings
from app.llm.providers import provider_status
from app.memory.models import Conversation, Escalation, Message, Run, RunEvent
from app.runtime import RunService

logger = logging.getLogger("agent_orch.api")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _ts(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def run_out(run: Run) -> dict[str, Any]:
    return {
        "id": run.id,
        "conversation_id": run.conversation_id,
        "task": run.task,
        "status": run.status,
        "provider": run.provider,
        "final_answer": run.final_answer,
        "error": run.error,
        "created_at": _ts(run.created_at),
        "started_at": _ts(run.started_at),
        "finished_at": _ts(run.finished_at),
        "output_data": run.output_data or {},
    }


def message_out(msg: Message) -> dict[str, Any]:
    return {
        "id": msg.id,
        "role": msg.role,
        "name": msg.name,
        "content": msg.content,
        "tool_calls": msg.tool_calls,
        "seq": msg.seq,
    }


def event_out(ev: RunEvent) -> dict[str, Any]:
    return {
        "id": ev.id,
        "type": ev.type,
        "actor": ev.actor,
        "payload": ev.payload or {},
        "created_at": _ts(ev.created_at),
    }


def escalation_out(esc: Escalation) -> dict[str, Any]:
    return {
        "id": esc.id,
        "run_id": esc.run_id,
        "kind": esc.kind,
        "status": esc.status,
        "reason": esc.reason,
        "question": esc.question,
        "context": esc.context or {},
        "decision": esc.decision,
        "created_at": _ts(esc.created_at),
        "resolved_at": _ts(esc.resolved_at),
    }


def create_app(service: RunService | None = None) -> FastAPI:
    settings = get_settings()
    service = service or RunService(settings=settings)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Supervisor-led multi-agent orchestration with tool use, persistent "
            "memory (PostgreSQL + ChromaDB) and human-in-the-loop escalation."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.service = service

    # ── Health & introspection ──────────────────────────────────────────────
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        checks: dict[str, Any] = {}
        try:
            with service.database.session() as s:
                s.execute(sql_text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks["database"] = f"error: {exc}"
        try:
            checks["vector_memory"] = f"ok ({service.vector.count()} memories)"
        except Exception as exc:  # noqa: BLE001
            checks["vector_memory"] = f"error: {exc}"
        if settings.queue_mode == "celery":
            try:
                import redis as redis_lib

                redis_lib.Redis.from_url(settings.redis_url).ping()
                checks["redis"] = "ok"
            except Exception as exc:  # noqa: BLE001
                checks["redis"] = f"error: {exc}"
        else:
            checks["redis"] = "not used (QUEUE_MODE=inline)"
        checks["llm"] = provider_status()
        return {
            "app": settings.app_name,
            "version": __version__,
            "environment": settings.environment,
            "queue_mode": settings.queue_mode,
            "checks": checks,
        }

    # ── Conversations ───────────────────────────────────────────────────────
    @app.post("/api/conversations")
    def create_conversation(body: ConversationCreate) -> dict[str, Any]:
        conv = service.create_conversation(body.title)
        return {"id": conv.id, "title": conv.title, "created_at": _ts(conv.created_at)}

    @app.get("/api/conversations")
    def list_conversations(limit: int = 50) -> list[dict[str, Any]]:
        return [
            {"id": c.id, "title": c.title, "created_at": _ts(c.created_at)}
            for c in service.store.list_conversations(limit=limit)
        ]

    @app.get("/api/conversations/{conversation_id}/messages")
    def conversation_messages(conversation_id: str) -> list[dict[str, Any]]:
        return [message_out(m) for m in service.store.get_messages(conversation_id=conversation_id)]

    # ── Runs ────────────────────────────────────────────────────────────────
    @app.post("/api/runs")
    async def create_run(body: RunCreate) -> dict[str, Any]:
        if body.mode == "sync":
            result = await asyncio.to_thread(
                service.submit_run,
                body.task,
                conversation_id=body.conversation_id,
                mode="sync",
                **body.metadata,
            )
            return run_out(result)
        run = service.create_run(body.task, conversation_id=body.conversation_id, **body.metadata)
        if settings.queue_mode == "celery":
            from app.queue.tasks import execute_run_task

            execute_run_task.delay(run.id)
        else:
            asyncio.get_running_loop().run_in_executor(None, service.execute_run, run.id)
        return run_out(service.store.get_run(run.id) or run)

    @app.get("/api/runs")
    def list_runs(limit: int = 50, conversation_id: str | None = None) -> list[dict[str, Any]]:
        return [
            run_out(r)
            for r in service.store.list_runs(limit=limit, conversation_id=conversation_id)
        ]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = service.store.get_run(run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        payload = run_out(run)
        payload["messages"] = [message_out(m) for m in service.store.get_messages(run_id=run_id)]
        payload["events"] = [event_out(e) for e in service.store.list_events(run_id)]
        pending = service.store.get_pending_escalation(run_id)
        payload["pending_escalation"] = escalation_out(pending) if pending else None
        return payload

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str, since_id: int = 0) -> list[dict[str, Any]]:
        return [event_out(e) for e in service.store.list_events(run_id, since_id=since_id)]

    @app.get("/api/runs/{run_id}/stream")
    def run_stream(run_id: str, since_id: int = 0) -> StreamingResponse:
        """Server-sent events feed of the run's activity timeline."""

        def gen() -> Iterator[str]:
            cursor = since_id
            for _ in range(1200):  # ~10 minutes at 0.5s polling
                events = service.store.list_events(run_id, since_id=cursor)
                for ev in events:
                    cursor = ev.id
                    yield f"data: {json.dumps(event_out(ev))}\n\n"
                run = service.store.get_run(run_id)
                if run is not None and run.status in ("completed", "failed", "cancelled") and not events:
                    yield f"data: {json.dumps({'type': 'stream.end', 'status': run.status})}\n\n"
                    return
                if run is not None and run.status == "awaiting_human" and not events:
                    yield f"data: {json.dumps({'type': 'stream.awaiting_human'})}\n\n"
                    return

                time.sleep(0.5)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/runs/{run_id}/resume")
    async def resume_run(run_id: str, body: EscalationDecision) -> dict[str, Any]:
        run = service.store.get_run(run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        if body.decision not in ("approve", "reject", "modify"):
            raise HTTPException(400, "decision must be approve | reject | modify")
        result = await asyncio.to_thread(service.resume_run, run_id, body.as_resume_value())
        return result

    # ── Escalations ─────────────────────────────────────────────────────────
    @app.get("/api/escalations")
    def list_escalations(status: str | None = Query(None), limit: int = 50) -> list[dict[str, Any]]:
        return [
            escalation_out(e) for e in service.store.list_escalations(status=status, limit=limit)
        ]

    @app.get("/api/escalations/{escalation_id}")
    def get_escalation(escalation_id: str) -> dict[str, Any]:
        esc = service.store.get_escalation(escalation_id)
        if esc is None:
            raise HTTPException(404, "escalation not found")
        return escalation_out(esc)

    @app.post("/api/escalations/{escalation_id}/resolve")
    async def resolve_escalation(escalation_id: str, body: EscalationDecision) -> dict[str, Any]:
        esc = service.store.get_escalation(escalation_id)
        if esc is None:
            raise HTTPException(404, "escalation not found")
        if esc.status != "pending":
            raise HTTPException(409, f"escalation already {esc.status}")
        if body.decision not in ("approve", "reject", "modify"):
            raise HTTPException(400, "decision must be approve | reject | modify")
        result = await asyncio.to_thread(service.resume_run, esc.run_id, body.as_resume_value())
        return {"escalation": escalation_out(service.store.get_escalation(escalation_id)), "run": result}

    # ── Memory ──────────────────────────────────────────────────────────────
    @app.get("/api/memory/search")
    def memory_search(q: str = Query(..., min_length=1), k: int = 5) -> dict[str, Any]:
        hits = service.vector.search(q, k=k)
        return {"query": q, "hits": [h.as_dict() for h in hits]}

    @app.post("/api/memory/search")
    def memory_search_post(body: MemorySearch) -> dict[str, Any]:
        hits = service.vector.search(body.q, k=body.k)
        return {"query": body.q, "hits": [h.as_dict() for h in hits]}

    @app.get("/api/memory/records")
    def memory_records(kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {
                "id": r.id,
                "kind": r.kind,
                "content": r.content,
                "metadata": r.metadata_ or {},
                "created_at": _ts(r.created_at),
            }
            for r in service.store.list_memory_records(kind=kind, limit=limit)
        ]

    # ── Web consoles ─────────────────────────────────────────────────────────
    # Zero-build console at /console (and / for convenience).
    if STATIC_DIR.exists():
        app.mount("/console", StaticFiles(directory=str(STATIC_DIR), html=True), name="console")

    # React console (production build from frontend-react/, served under /react).
    react_dist = Path(__file__).resolve().parent.parent.parent.parent / "frontend-react" / "dist"
    if not react_dist.exists():
        # Docker layout: /srv/frontend-react/dist
        react_dist = Path("/srv/frontend-react/dist")
    if react_dist.exists():
        app.mount("/react", StaticFiles(directory=str(react_dist), html=True), name="react-console")

    @app.get("/")
    def index() -> FileResponse:
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(404, "console not found")

    return app


def main() -> None:  # pragma: no cover - uvicorn entry
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.api.app:create_app", factory=True, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":  # pragma: no cover
    main()
