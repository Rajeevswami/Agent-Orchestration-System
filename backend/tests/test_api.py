"""HTTP API tests: FastAPI app with injected RunService + scripted LLMs."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.llm.scripted import register_script


@pytest.fixture()
def client(service):
    app = create_app(service=service)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["checks"]["database"] == "ok"
    assert data["checks"]["llm"]["provider"] == "scripted"
    assert data["checks"]["llm"]["model"] == "scripted (deterministic)"


def test_console_and_docs_served(client):
    assert client.get("/").status_code == 200
    assert "Agent Orchestration Console" in client.get("/").text
    assert client.get("/docs").status_code == 200


def test_sync_run_completes(client):
    register_script("supervisor", [
        '{"action": "writer", "reasoning": "quick", "instructions": "answer", "confidence": 0.9}',
        '{"action": "FINISH", "reasoning": "done", "instructions": "", "confidence": 0.95}',
    ])
    register_script("writer", ["The answer is 7. SUMMARY: done."])

    resp = client.post("/api/runs", json={"task": "what is 3+4?", "mode": "sync"})
    assert resp.status_code == 200
    run = resp.json()
    assert run["status"] == "completed"
    assert "7" in run["final_answer"]

    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["output_data"]["route_history"] == ["writer", "FINISH"]
    assert any(m["role"] == "user" for m in detail["messages"])
    assert detail["events"]  # activity timeline populated


def test_escalation_flow_over_http(client):
    register_script("supervisor", [
        # turn 1 — consumed by the policy gate (wire transfer) -> escalate
        '{"action": "researcher", "reasoning": "context", "instructions": "look up vendor", "confidence": 0.9}',
        # turn 2 — after human approval, run the planned research
        '{"action": "researcher", "reasoning": "approved, proceed", "instructions": "look up vendor", "confidence": 0.9}',
        '{"action": "FINISH", "reasoning": "done", "instructions": "", "confidence": 0.95}',
    ])
    register_script("researcher", ["Vendor terms found. SUMMARY: net 30."])

    run = client.post(
        "/api/runs", json={"task": "Send a wire transfer of $500 to Acme", "mode": "sync"}
    ).json()
    assert run["status"] == "awaiting_human"

    pending = client.get("/api/escalations?status=pending").json()
    assert len(pending) == 1
    esc = pending[0]
    assert esc["run_id"] == run["id"]
    assert esc["status"] == "pending"
    assert "financial_transaction" in esc["reason"]

    resolved = client.post(
        f"/api/escalations/{esc['id']}/resolve",
        json={"decision": "approve", "comment": "policy ok"},
    ).json()
    assert resolved["run"]["status"] == "completed"
    assert resolved["escalation"]["status"] == "approved"
    assert "net 30" in resolved["run"]["final_answer"]

    # Double-resolve is rejected.
    again = client.post(
        f"/api/escalations/{esc['id']}/resolve", json={"decision": "approve"}
    )
    assert again.status_code == 409


def test_resume_endpoint(client):
    register_script("supervisor", [
        '{"action": "writer", "reasoning": "x", "instructions": "draft", "confidence": 0.2}',
        '{"action": "FINISH", "reasoning": "done", "instructions": "", "confidence": 0.95}',
    ])
    register_script("writer", ["Draft. SUMMARY: ok."])

    run = client.post("/api/runs", json={"task": "draft a note", "mode": "sync"}).json()
    assert run["status"] == "awaiting_human"  # low confidence

    result = client.post(
        f"/api/runs/{run['id']}/resume", json={"decision": "approve", "comment": "go"}
    ).json()
    assert result["status"] == "completed"


def test_conversations_and_memory_endpoints(client):
    conv = client.post("/api/conversations", json={"title": "Test"}).json()
    assert conv["id"]

    register_script("supervisor", [
        '{"action": "FINISH", "reasoning": "trivial", "instructions": "", "confidence": 0.95}',
    ])
    run = client.post(
        "/api/runs",
        json={"task": "remember the launch date is March 12", "mode": "sync", "conversation_id": conv["id"]},
    ).json()

    msgs = client.get(f"/api/conversations/{conv['id']}/messages").json()
    assert msgs and msgs[0]["role"] == "user"

    # Run consolidation writes into semantic memory.
    hits = client.get("/api/memory/search", params={"q": "launch date"}).json()
    assert hits["hits"], "consolidated memory should be searchable"

    records = client.get("/api/memory/records").json()
    assert records

    assert client.get("/api/conversations").json()[0]["id"] == conv["id"]


def test_run_events_endpoint(client):
    register_script("supervisor", [
        '{"action": "writer", "reasoning": "x", "instructions": "draft", "confidence": 0.9}',
        '{"action": "FINISH", "reasoning": "done", "instructions": "", "confidence": 0.95}',
    ])
    register_script("writer", ["Hi. SUMMARY: ok."])
    run = client.post("/api/runs", json={"task": "say hi", "mode": "sync"}).json()

    events = client.get(f"/api/runs/{run['id']}/events").json()
    assert events
    types = [e["type"] for e in events]
    assert "supervisor.route" in types and "worker.completed" in types

    # since_id pagination
    tail = client.get(f"/api/runs/{run['id']}/events", params={"since_id": events[0]["id"]}).json()
    assert all(e["id"] > events[0]["id"] for e in tail)


def test_validation_errors(client):
    assert client.post("/api/runs", json={"task": ""}).status_code == 422
    assert client.get("/api/runs/does-not-exist").status_code == 404
    assert client.post(
        "/api/runs/x/resume", json={"decision": "maybe"}
    ).status_code in (400, 404)
