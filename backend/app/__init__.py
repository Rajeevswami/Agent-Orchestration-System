"""Agent Orchestration System — supervisor-led multi-agent runtime.

The package is organised as:

* :mod:`app.config`          — settings (env-driven)
* :mod:`app.llm`             — provider router, prompts, scripted models
* :mod:`app.memory`          — PostgreSQL/SQLite store + ChromaDB vector memory
* :mod:`app.agents`          — tools and worker agents
* :mod:`app.orchestrator`    — LangGraph supervisor graph, escalation policy
* :mod:`app.queue`           — Redis + Celery task bridge
* :mod:`app.api`             — FastAPI service + web console
* :mod:`app.runtime`         — run service tying everything together
"""

__version__ = "1.0.0"
