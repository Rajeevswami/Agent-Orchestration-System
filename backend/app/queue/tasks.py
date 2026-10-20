"""Celery tasks wrapping :class:`app.runtime.RunService`.

With ``QUEUE_MODE=inline`` these are never dispatched — the RunService executes
in-process — so dev machines and CI need no Redis. In production, the API
enqueues here and one or more workers do the (potentially long) LLM work.
"""

from __future__ import annotations

from typing import Any

from app.queue.celery_app import celery_app


@celery_app.task(bind=True, name="agent.execute_run", max_retries=2)
def execute_run_task(self, run_id: str) -> dict[str, Any]:
    from app.runtime import RunService

    try:
        service = RunService()
        return service.execute_run(run_id)
    except Exception as exc:  # noqa: BLE001
        raise self.retry(exc=exc, countdown=10) from exc


@celery_app.task(bind=True, name="agent.resume_run", max_retries=2)
def resume_run_task(self, run_id: str, decision: dict[str, Any]) -> dict[str, Any]:
    from app.runtime import RunService

    try:
        service = RunService()
        return service.resume_run(run_id, decision)
    except Exception as exc:  # noqa: BLE001
        raise self.retry(exc=exc, countdown=10) from exc


@celery_app.task(name="agent.consolidate_memory")
def consolidate_memory_task(run_id: str) -> bool:
    """Re-run memory consolidation for a run (operational hook)."""
    from app.runtime import RunService

    service = RunService()
    run = service.store.get_run(run_id)
    if run is None:
        return False
    service._consolidate_memory(run, {"final_answer": run.final_answer, "supervisor_notes": []})
    return True
