"""Queue subsystem: Celery tasks backed by Redis."""

from app.queue.celery_app import celery_app

__all__ = ["celery_app"]
