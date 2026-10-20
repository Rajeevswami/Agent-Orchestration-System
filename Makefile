PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
export PYTHONPATH := backend

.PHONY: help venv install test demo run-api run-worker run-ui up down logs

help:
	@echo "Agent Orchestration System"
	@echo "  make install    create venv + install dependencies"
	@echo "  make test       run the test-suite (offline, no API keys)"
	@echo "  make demo       scripted end-to-end demo (offline)"
	@echo "  make run-api    start FastAPI + web console on :8000"
	@echo "  make run-worker start a Celery worker (needs Redis)"
	@echo "  make run-ui     start the Streamlit frontend on :8501"
	@echo "  make up         docker compose up (postgres, redis, chroma, api, worker, ui)"
	@echo "  make down       docker compose down"

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	cd backend && ../.venv/bin/python -m pytest

demo:
	cd backend && ../.venv/bin/python -m app.cli demo

run-api:
	cd backend && ../.venv/bin/python -m app.cli serve

run-worker:
	cd backend && celery -A app.queue.celery_app:celery_app worker --loglevel=info

run-ui:
	.venv/bin/streamlit run frontend/streamlit_app.py

up:
	docker compose up --build

down:
	docker compose down -v

logs:
	docker compose logs -f api worker
