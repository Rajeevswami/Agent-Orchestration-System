"""Run context shared by graph nodes and tools (memory handles + event feed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.memory.store import RelationalStore
from app.memory.vector import VectorMemory


@dataclass
class RunContext:
    """Everything a node needs to interact with the outside world."""

    run_id: str
    conversation_id: str
    settings: Settings
    store: RelationalStore
    vector: VectorMemory
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, type: str, actor: str = "", payload: dict[str, Any] | None = None) -> None:
        """Broadcast an activity event (persisted for the live UI timeline)."""
        payload = payload or {}
        event = {"type": type, "actor": actor, "payload": payload}
        self.events.append(event)
        try:
            self.store.add_event(self.run_id, type, actor, payload)
        except Exception:  # events must never break the run
            pass
