"""Environment-driven configuration for the orchestration system."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime knobs, loadable from environment variables or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── General ──────────────────────────────────────────────────────────────
    app_name: str = "Agent Orchestration System"
    environment: str = "development"

    # ── LLM providers ────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic", "scripted"] = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    # Optional per-role model overrides (fall back to the provider default).
    supervisor_model: str = ""
    researcher_model: str = ""
    analyst_model: str = ""
    writer_model: str = ""

    temperature: float = 0.2
    max_tokens: int = 2048

    # ── Memory ───────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./data/agent_orchestrator.db"

    chroma_mode: Literal["persistent", "ephemeral", "http"] = "persistent"
    chroma_path: str = "./data/chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    embedding_provider: Literal["hashing", "openai"] = "hashing"
    embedding_model: str = "text-embedding-3-small"
    memory_top_k: int = 5
    memory_autosave: bool = True

    # ── Queue ────────────────────────────────────────────────────────────────
    queue_mode: Literal["inline", "celery"] = "inline"
    redis_url: str = "redis://localhost:6379/0"

    # ── Orchestration policy ─────────────────────────────────────────────────
    max_supervisor_iterations: int = 8
    max_tool_steps: int = 5
    min_confidence: float = 0.55
    escalation_on_low_confidence: bool = True
    require_final_approval: bool = False

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"

    def model_for(self, role: str) -> str:
        """Resolve the configured model name for a role (supervisor/worker)."""
        override = {
            "supervisor": self.supervisor_model,
            "researcher": self.researcher_model,
            "analyst": self.analyst_model,
            "writer": self.writer_model,
        }.get(role, "")
        if override:
            return override
        return self.openai_model if self.llm_provider == "openai" else self.anthropic_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance (call ``get_settings.cache_clear()`` in tests)."""
    return Settings()
