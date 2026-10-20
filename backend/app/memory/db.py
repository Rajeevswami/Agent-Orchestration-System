"""Engine / session management for the relational memory store."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.memory.models import Base


def _ensure_sqlite_dir(url: str) -> None:
    """Create the parent directory for SQLite database files."""
    if url.startswith("sqlite:///") and not url.endswith("sqlite:///:memory:"):
        path = url.replace("sqlite:///", "", 1)
        if path and path != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


class Database:
    """Small wrapper around an SQLAlchemy engine + session factory."""

    def __init__(self, url: str):
        self.url = url
        _ensure_sqlite_dir(url)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(
            url, future=True, pool_pre_ping=True, connect_args=connect_args
        )
        self.session_factory = sessionmaker(
            bind=self.engine, expire_on_commit=False, future=True
        )

    def init_db(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session: Session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
