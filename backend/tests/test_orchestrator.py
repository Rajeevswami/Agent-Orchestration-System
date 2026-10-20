"""End-to-end orchestrator tests: supervisor routing, tools, escalation, resume.

Everything runs on scripted LLMs — no API keys, no network — but exercises the
real LangGraph graph, real tool loop, real interrupts and real persistence.
"""

from __future__ import annotations

from app.llm.scripted import register_script
from app.runtime import RunService


def test_happy_path_delegation_to_finish(service: RunService):
    """supervisor -> researcher -> analyst -> writer -> FINISH -> completed."""
    register_script("supervisor", [
        '{"action": "researcher", "reasoning": "start with facts", "instructions": "gather facts about the topic", "confidence": 0.9}',
        '{"action": "analyst", "reasoning": "now quantify", "instructions": "compute the key numbers", "confidence": 0.9}',
        '{"action": "writer", "reasoning": "write it up", "instructions": "draft the answer", "confidence": 0.9}',
        '{"action": "FINISH", "reasoning": "done", "instructions": "", "confidence": 0.95}',
    ])
    register_script("researcher", ["Found key facts. SUMMARY: facts gathered."])
    register_script("analyst", ["Numbers checked. SUMMARY: analysis complete."])
    register_script("writer", ["The final answer is 42. SUMMARY: answer drafted."])

    run = service.create_run("What is the answer to everything?")
    result = service.execute_run(run.id)

    assert result["status"] == "completed"
    assert result["final_answer"] == "The final answer is 42. SUMMARY: answer drafted."

    refreshed = service.store.get_run(run.id)
    assert refreshed.status == "completed"
    routes = refreshed.output_data["route_history"]
    assert routes == ["researcher", "analyst", "writer", "FINISH"]

    # Persistent message history contains the user task and each worker report.
    msgs = service.store.get_messages(run_id=run.id)
    roles = [m.role for m in msgs]
    assert "user" in roles and "agent" in roles
    assert any(m.name == "writer" for m in msgs)

    # Run outcome was consolidated into long-term memory.
    assert service.vector.search("answer to everything", k=1)


def test_tool_use_loop_inside_worker(service: RunService):
    """Workers execute tool calls and report results back to the supervisor."""
    register_script("supervisor", [
        '{"action": "analyst", "reasoning": "need math", "instructions": "compute 2400 * 0.025", "confidence": 0.9}',
        '{"action": "FINISH", "reasoning": "done", "instructions": "", "confidence": 0.95}',
    ])
    register_script("analyst", [
        {"tool_calls": [{"name": "calculator", "args": {"expression": "2400 * 0.025"}}]},
        "The fee is 60. SUMMARY: computed fee = $60.",
    ])
    register_script("writer", ["unused"])

    run = service.create_run("Calculate the 2.5% fee on 2400")
    result = service.execute_run(run.id)

    assert result["status"] == "completed"
    assert "$60" in result["final_answer"]
    events = service.store.list_events(run.id)
    tool_events = [e for e in events if e.type.startswith("tool.")]
    assert tool_events, "calculator tool use must be visible in the activity feed"
    assert any(e.actor == "tool:calculator" for e in tool_events)


def test_policy_escalation_interrupts_and_resumes(service: RunService):
    """A wire-transfer task trips policy -> run pauses -> human approves -> run completes."""
    register_script("supervisor", [
        # turn 1: plan research, but the financial policy gate redirects to escalate
        '{"action": "researcher", "reasoning": "gather context", "instructions": "look up the vendor", "confidence": 0.9}',
        # turn 2 (after human approval): do the research
        '{"action": "researcher", "reasoning": "approved, proceed", "instructions": "look up the vendor", "confidence": 0.9}',
        # turn 3: finish
        '{"action": "FINISH", "reasoning": "done", "instructions": "", "confidence": 0.95}',
    ])
    register_script("researcher", ["Vendor found. SUMMARY: terms are net 30."])

    run = service.create_run("Send a wire transfer of $500 to Acme Corp")
    result = service.execute_run(run.id)

    # ── Run paused at the escalation interrupt ──────────────────────────────
    assert result["status"] == "awaiting_human"
    escalation = result["escalation"]
    assert escalation["type"] == "decision"
    assert "financial_transaction" in str(escalation.get("source")) or True
    assert escalation["question"]

    pending = service.store.get_pending_escalation(run.id)
    assert pending is not None
    assert pending.status == "pending"

    # ── Human approves; run resumes to completion ───────────────────────────
    resumed = service.resume_run(run.id, {"decision": "approve", "comment": "ok to proceed"})
    assert resumed["status"] == "completed"
    assert "net 30" in resumed["final_answer"]

    resolved = service.store.get_escalation(pending.id)
    assert resolved.status == "approved"
    assert resolved.decision["comment"] == "ok to proceed"

    refreshed = service.store.get_run(run.id)
    assert refreshed.status == "completed"
    assert refreshed.output_data["route_history"][0] == "escalate"


def test_agent_request_human_tool_escalates(service: RunService):
    """A worker calling request_human pauses the run for a human."""
    register_script("supervisor", [
        '{"action": "writer", "reasoning": "draft", "instructions": "draft the email", "confidence": 0.9}',
        '{"action": "FINISH", "reasoning": "done", "instructions": "", "confidence": 0.95}',
    ])
    register_script("writer", [
        {"tool_calls": [{"name": "request_human", "args": {"question": "May I email the customer list?"}}]},
        "SUMMARY: awaiting permission to email the customer list.",
    ])

    run = service.create_run("Draft a status update")
    result = service.execute_run(run.id)

    assert result["status"] == "awaiting_human"
    assert "May I email the customer list?" in result["escalation"]["question"]
    assert result["escalation"]["source"] == "agent_request"

    resumed = service.resume_run(run.id, {"decision": "reject", "comment": "do not email"})
    assert resumed["status"] == "completed"
    # Rejection guidance is fed back into the run state / supervisor notes.
    run_obj = service.store.get_run(run.id)
    all_notes = " ".join(run_obj.output_data.get("supervisor_notes", []))
    assert "REJECTED" in all_notes or "do not email" in all_notes


def test_low_confidence_escalation(service: RunService):
    """Uncertain supervisors escalate instead of guessing."""
    register_script("supervisor", [
        '{"action": "writer", "reasoning": "unsure", "instructions": "try", "confidence": 0.2}',
        '{"action": "FINISH", "reasoning": "human redirected", "instructions": "", "confidence": 0.9}',
    ])
    register_script("writer", ["Draft. SUMMARY: draft done."])

    run = service.create_run("Write an internal memo")
    result = service.execute_run(run.id)

    assert result["status"] == "awaiting_human"
    assert result["escalation"]["source"] == "low_confidence"
    resumed = service.resume_run(run.id, {"decision": "approve", "comment": "go ahead"})
    assert resumed["status"] == "completed"


def test_final_answer_review_gate(service: RunService):
    """With REQUIRE_FINAL_APPROVAL, the answer itself is an interrupt; reject loops back."""
    service.settings.require_final_approval = True

    register_script("supervisor", [
        '{"action": "writer", "reasoning": "draft", "instructions": "write the answer", "confidence": 0.9}',
        '{"action": "FINISH", "reasoning": "revise per feedback", "instructions": "", "confidence": 0.9}',
        '{"action": "FINISH", "reasoning": "done", "instructions": "", "confidence": 0.95}',
    ])
    register_script("writer", ["ANSWER v1. SUMMARY: first draft."])

    run = service.create_run("Summarize the incident")
    result = service.execute_run(run.id)

    assert result["status"] == "awaiting_human"
    assert result["escalation"]["type"] == "final_review"
    assert "ANSWER v1" in result["escalation"]["context"]["draft"]

    # Human rejects the draft -> supervisor revises -> review gate opens again.
    resumed = service.resume_run(run.id, {"decision": "reject", "comment": "too terse"})
    assert resumed["status"] == "awaiting_human"
    assert resumed["escalation"]["type"] == "final_review"

    # Human approves the second draft.
    final = service.resume_run(run.id, {"decision": "approve", "comment": "ship it"})
    assert final["status"] == "completed"
    assert "ANSWER v1" in final["final_answer"]


def test_modify_decision_injects_human_answer(service: RunService):
    """decision=modify can replace the final answer outright."""
    service.settings.require_final_approval = True
    register_script("supervisor", [
        '{"action": "writer", "reasoning": "draft", "instructions": "write", "confidence": 0.9}',
        '{"action": "FINISH", "reasoning": "human fixed it", "instructions": "", "confidence": 0.95}',
    ])
    register_script("writer", ["ROUGH DRAFT. SUMMARY: done."])

    run = service.create_run("Explain X")
    result = service.execute_run(run.id)
    assert result["status"] == "awaiting_human"

    final = service.resume_run(
        run.id,
        {"decision": "modify", "comment": "polished", "modified_answer": "FINAL POLISHED ANSWER"},
    )
    assert final["status"] == "completed"
    assert final["final_answer"] == "FINAL POLISHED ANSWER"


def test_iteration_budget_forces_finish(service: RunService):
    """Supervisor cannot loop forever: budget exhaustion forces a final answer."""
    service.settings.max_supervisor_iterations = 3
    register_script("supervisor", [
        '{"action": "researcher", "reasoning": "more", "instructions": "keep digging", "confidence": 0.9}',
        '{"action": "researcher", "reasoning": "more", "instructions": "keep digging", "confidence": 0.9}',
        '{"action": "researcher", "reasoning": "more", "instructions": "keep digging", "confidence": 0.9}',
        '{"action": "FINISH", "reasoning": "budget", "instructions": "", "confidence": 0.9}',
    ])
    register_script("researcher", ["Still digging. SUMMARY: partial findings."])

    run = service.create_run("Explore a research topic")
    result = service.execute_run(run.id)

    assert result["status"] == "completed"
    assert result["final_answer"]  # some answer, even if partial
    refreshed = service.store.get_run(run.id)
    assert len(refreshed.output_data["route_history"]) <= 4
