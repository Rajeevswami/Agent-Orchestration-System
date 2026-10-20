"""Shared fixtures: isolated stores + scripted LLMs (no API keys, no network)."""

from __future__ import annotations

import os

import pytest

# Point every test at an ephemeral SQLite DB + fresh Chroma namespace.
os.environ.setdefault("LLM_PROVIDER", "scripted")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("CHROMA_MODE", "persistent")
os.environ.setdefault("CHROMA_PATH", "/tmp/agent_orch_test_chroma")
os.environ.setdefault("QUEUE_MODE", "inline")
os.environ.setdefault("EMBEDDING_PROVIDER", "hashing")

from app.config import Settings  # noqa: E402
from app.memory.db import Database  # noqa: E402
from app.memory.store import RelationalStore  # noqa: E402
from app.memory.vector import VectorMemory  # noqa: E402
from app.runtime import RunService  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_script_registry():
    from app.llm.scripted import REGISTRY

    REGISTRY.clear()
    yield
    REGISTRY.clear()


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        llm_provider="scripted",
        database_url=f"sqlite:///{tmp_path}/test.db",
        chroma_mode="persistent",
        chroma_path=str(tmp_path / "chroma"),
        queue_mode="inline",
        embedding_provider="hashing",
        require_final_approval=False,
    )


@pytest.fixture()
def store(settings) -> RelationalStore:
    db = Database(settings.database_url)
    db.init_db()
    return RelationalStore(db)


@pytest.fixture()
def vector(settings) -> VectorMemory:
    return VectorMemory(settings)


@pytest.fixture()
def service(settings) -> RunService:
    from langgraph.checkpoint.memory import InMemorySaver

    return RunService(settings=settings, checkpointer=InMemorySaver())
