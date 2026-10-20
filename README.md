# 🤖 Agent Orchestration System

A production-shaped **multi-agent orchestration platform** with tool use, persistent
memory, and human-in-the-loop escalation — not a single-agent demo.

A **supervisor** agent plans and delegates to specialised **researcher / analyst /
writer** workers on a [LangGraph](https://github.com/langchain-ai/langgraph) state
graph. Every decision, message and tool call is persisted (PostgreSQL), every agent
can read/write long-term semantic memory (ChromaDB), long runs execute on a
**Redis + Celery** queue, and **policy/ambiguity/uncertainty pauses the run for a
human decision** that resumes it mid-flight.

```
                        ┌──────────────────────────────────────────────┐
                        │                 FastAPI + Console            │
                        │        (REST · SSE · web console · CLI)      │
                        └──────────────┬───────────────────────────────┘
                                       │
       ┌───────────────────────────────┼───────────────────────────────────┐
       │                  LangGraph orchestration graph                    │
       │                                                                   │
       │   START → retrieve memories → SUPERVISOR ──┬─→ researcher ─┐      │
       │          (ChromaDB recall)        │        ├─→ analyst    ─┼──┐   │
       │                                   │        ├─→ writer     ─┘  │   │
       │                                   │        └─→ escalate ───┐  │   │
       │                                   ▼                        │  │   │
       │                               finalize ──→ [review] ──→ END│  │   │
       │                                   ▲    (optional human gate)   │   │
       │                                   └───── human decision ───────┘   │
       └───────────────────────────────────────────────────────────────────┘
                 │                    │                       │
          ┌──────┴──────┐      ┌──────┴──────┐        ┌───────┴────────┐
          │ PostgreSQL  │      │  ChromaDB   │        │ Redis + Celery │
          │ runs, msgs, │      │  semantic   │        │  async runs,   │
          │ events,     │      │  long-term  │        │  resume tasks  │
          │ escalations │      │  memory     │        │                │
          └─────────────┘      └─────────────┘        └────────────────┘
```

## Why this is different

| Typical agent demo | This system |
|---|---|
| One agent, one prompt | **Supervisor delegation** with specialist workers and routing history |
| Chat history in RAM | **Durable state**: every run, message, tool call and decision in PostgreSQL |
| "Memory" = stuffing the prompt | **Semantic memory** in ChromaDB, auto-consolidated per run, recalled per task |
| Fire-and-forget | **Redis + Celery queue** for background execution at scale |
| Fully autonomous, unsupervised | **Human-in-the-loop interrupts**: policy engine + agent `request_human` tool + optional final-answer review, resumed via `Command(resume=...)` |
| Hard-coded happy path | **Policy engine** (financial ops, deletions, PII, credentials, irreversibility), confidence thresholds, iteration budgets |

## Features

- **🧠 Supervisor-led orchestration** (LangGraph) — plans, delegates, tracks route
  history, forces completion on iteration budget.
- **🛠 Tool use** — `web_search`, safe AST `calculator`, `retrieve_memory`,
  `save_memory`, `request_human`, `current_datetime`; per-tool activity events.
- **🐘 Persistent memory** — PostgreSQL/SQLite for episodic memory (runs, messages,
  events, escalations) + ChromaDB semantic memory with pluggable embeddings
  (`hashing` = fully offline, or OpenAI embeddings).
- **🛑 Human-in-the-loop** — runs pause at LangGraph interrupts with a durable
  `Escalation` record; humans **approve / reject / modify** from the API, web
  console or Streamlit; the graph resumes with the decision folded into state.
- **📨 Queue** — inline executor for dev, Celery + Redis for production.
- **🔀 Providers** — OpenAI *and* Anthropic behind one router (per-role model
  overrides), plus a **scripted provider** for offline demos and CI.
- **🖥 Frontends** — built-in web console (zero-build, served at `/`) and a
  Streamlit app (`frontend/streamlit_app.py`).
- **🔌 REST + SSE API** — create runs, stream activity, resolve escalations,
  search memory (`/api/health`, `/api/runs`, `/api/escalations`, `/api/memory/*`).
- **🧪 Tested offline** — 28 tests exercise the real graph, real interrupts and
  real persistence with scripted LLMs (no API keys needed).

## Quickstart

### 1. Local (no Docker, SQLite + inline queue)

```bash
make install
make test          # offline test-suite
make demo          # scripted end-to-end demo: escalation → tools → final answer
make run-api       # API + web console on http://localhost:8000
make run-ui        # Streamlit frontend on http://localhost:8501
```

The demo needs **no API keys** — it drives the real graph with scripted models and
shows the full loop: policy gate → human approval → research tools → analyst
calculator → writer → memory consolidation.

### 2. With real LLMs

```bash
cp .env.example .env       # add OPENAI_API_KEY or ANTHROPIC_API_KEY
make run-api
```

```bash
curl -X POST http://localhost:8000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"task": "Research the impact of EU AI regulation on startups and draft a briefing", "mode": "sync"}'
```

### 3. Full stack (Docker)

```bash
docker compose up --build
```

Brings up **PostgreSQL 16 · Redis 7 · ChromaDB · FastAPI · Celery worker ·
Streamlit** — durable checkpointer + durable memory + background workers.

| Service | URL |
|---|---|
| Web console | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| ChromaDB | http://localhost:8001 |

## Human-in-the-loop escalation

A run is interrupted (and persisted) whenever:

1. **Policy rules** match — financial transactions, data deletion, PII/credential
   exposure, irreversible actions, external communications;
2. **Confidence** drops below `MIN_CONFIDENCE`;
3. A worker calls the **`request_human`** tool (permission / ambiguity);
4. The **supervisor** explicitly escalates (judgment calls);
5. The **iteration budget** is exhausted mid-plan;
6. `REQUIRE_FINAL_APPROVAL=true` — the **final answer itself** gets an approval gate.

Resolving an escalation posts a decision back into the suspended graph:

```bash
# list pending escalations
curl http://localhost:8000/api/escalations?status=pending

# approve / reject / modify — the run resumes mid-graph with the decision in state
curl -X POST http://localhost:8000/api/escalations/<id>/resolve \
  -H 'Content-Type: application/json' \
  -d '{"decision": "approve", "comment": "finance policy satisfied"}'
```

Decisions:
- **approve** — standing mandate; the team proceeds (policy gate will not re-fire
  for the same triggers);
- **reject** — feedback is injected and the supervisor revises the plan;
- **modify** — human guidance (or a replacement answer) is injected into state.

## Project structure

```
├── backend/app/
│   ├── config.py            # env-driven settings
│   ├── runtime.py           # RunService: execute / resume / persist / consolidate
│   ├── cli.py               # run · resume · demo · memory-search · serve
│   ├── demo.py              # offline scripted demo scenario
│   ├── llm/
│   │   ├── providers.py     # OpenAI | Anthropic | scripted router
│   │   ├── scripted.py      # deterministic models for tests/demos
│   │   └── prompts.py       # supervisor / worker / memory prompts
│   ├── memory/
│   │   ├── models.py        # SQLAlchemy: runs, messages, events, escalations…
│   │   ├── store.py         # relational episodic memory
│   │   └── vector.py        # ChromaDB semantic memory + embeddings
│   ├── agents/
│   │   ├── tools.py         # web_search · calculator · memory · request_human
│   │   └── workers.py       # bounded ReAct-style worker loops
│   ├── orchestrator/
│   │   ├── state.py         # shared LangGraph state
│   │   ├── supervisor.py    # graph assembly + supervisor routing
│   │   ├── escalation.py    # policy engine + escalation manager
│   │   └── context.py       # RunContext (memory + event feed)
│   ├── queue/               # Celery app + tasks (Redis broker)
│   ├── api/                 # FastAPI app + schemas (REST + SSE)
│   └── static/              # zero-build web console
├── backend/tests/           # offline pytest suite (scripted LLMs)
├── frontend/streamlit_app.py
├── docker-compose.yml       # postgres · redis · chroma · api · worker · ui
└── Makefile
```

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | DB / vector / Redis / LLM provider status |
| `POST` | `/api/runs` | Start a run (`mode=sync\|async`) |
| `GET` | `/api/runs/{id}` | Run with messages, events, pending escalation |
| `GET` | `/api/runs/{id}/stream` | SSE activity timeline |
| `POST` | `/api/runs/{id}/resume` | Feed a human decision into the run |
| `GET` | `/api/escalations?status=pending` | Human-in-the-loop queue |
| `POST` | `/api/escalations/{id}/resolve` | approve / reject / modify |
| `GET` | `/api/memory/search?q=` | Semantic memory search |
| `GET` | `/api/memory/records` | Durable fact log |

## Configuration

All knobs live in `.env` (see `.env.example`). Highlights:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `scripted` |
| `DATABASE_URL` | `sqlite:///./data/agent_orchestrator.db` | PostgreSQL in Docker |
| `CHROMA_MODE` | `persistent` | `persistent` \| `ephemeral` \| `http` |
| `EMBEDDING_PROVIDER` | `hashing` | `hashing` (offline) \| `openai` |
| `QUEUE_MODE` | `inline` | `inline` \| `celery` (needs Redis) |
| `MIN_CONFIDENCE` | `0.55` | Below this, the supervisor escalates |
| `MAX_SUPERVISOR_ITERATIONS` | `8` | Loop guard → forced finish |
| `REQUIRE_FINAL_APPROVAL` | `false` | Human gate on every final answer |

## Development

```bash
make test                 # full offline suite
cd backend && ../.venv/bin/python -m pytest tests/test_orchestrator.py -q   # graph tests
python -m app.cli run "task…"        # real run via CLI
python -m app.cli resume <run_id> approve
python -m app.cli memory-search "payment terms"
```

## Design notes

- **Graph, not chains.** The supervisor is a router over a `StateGraph`; workers are
  nodes with bounded internal tool loops. Everything is checkpointer-backed, so an
  interrupted run resumes at the exact node it paused on.
- **Escalation is a first-class state machine**, not a print statement: interrupts
  materialize as `Escalation` rows, decisions re-enter via `Command(resume=…)` and
  are recorded in run history for audit.
- **Memory has two tiers**: relational (authoritative episodic log) and vector
  (recallable semantic memory). Consolidation writes both after each run; agents
  can also `save_memory`/`retrieve_memory` mid-run.
- **Offline-first dev loop**: the scripted provider replays deterministic turns
  (including tool calls) so the entire orchestration stack is exercised in CI with
  zero keys and zero network.
