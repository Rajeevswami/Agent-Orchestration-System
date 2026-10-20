"""Provider router: one entry point for every chat model in the system.

Roles (``supervisor``, ``researcher``, ``analyst``, ``writer``) can point at
different models. The provider itself is selected with ``LLM_PROVIDER``:

* ``openai``    — LangChain ``ChatOpenAI``
* ``anthropic`` — LangChain ``ChatAnthropic``
* ``scripted``  — deterministic offline models (tests / ``--demo`` runs)
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings


def get_chat_model(role: str = "default", settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    provider = settings.llm_provider

    if provider == "scripted":
        from app.llm.scripted import REGISTRY

        return REGISTRY.get(role)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.model_for(role),
            api_key=settings.openai_api_key or None,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=60,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.model_for(role),
            api_key=settings.anthropic_api_key or None,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=60,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")


def provider_status(settings: Settings | None = None) -> dict:
    """Cheap introspection for the health endpoint / UI."""
    settings = settings or get_settings()
    configured = {
        "openai": bool(settings.openai_api_key),
        "anthropic": bool(settings.anthropic_api_key),
        "scripted": True,
    }
    if settings.llm_provider == "scripted":
        model = "scripted (deterministic)"
    else:
        model = settings.model_for("supervisor")
    return {
        "provider": settings.llm_provider,
        "configured": configured.get(settings.llm_provider, False),
        "model": model,
    }
