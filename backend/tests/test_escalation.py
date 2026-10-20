"""Escalation policy engine tests."""

from __future__ import annotations

from app.orchestrator.escalation import EscalationManager, PolicyEngine


def test_financial_action_triggers_policy(settings):
    engine = PolicyEngine(settings)
    check = engine.evaluate(task="Please wire transfer $500 to vendor X", confidence=0.9)
    assert check.needs_escalation
    assert check.source == "policy"
    assert "financial_transaction" in check.matched_rules


def test_deletion_and_pii_rules(settings):
    engine = PolicyEngine(settings)
    assert engine.evaluate(task="drop table users").needs_escalation
    assert engine.evaluate(plan="email me the credit card number of the customer").needs_escalation
    assert not engine.evaluate(task="summarize the quarterly report").needs_escalation


def test_low_confidence_escalation(settings):
    engine = PolicyEngine(settings)
    check = engine.evaluate(task="write a poem", confidence=0.2)
    assert check.needs_escalation
    assert check.source == "low_confidence"


def test_explicit_supervisor_escalation(settings):
    engine = PolicyEngine(settings)
    check = engine.evaluate(task="write a poem", confidence=0.99, explicit=True, explicit_question="Which tone?")
    assert check.needs_escalation
    assert check.source == "supervisor"
    assert check.question == "Which tone?"


def test_agent_tool_request_wins(settings):
    engine = PolicyEngine(settings)
    check = engine.evaluate(
        task="wire transfer money",  # would normally trip policy
        confidence=0.9,
        tool_request={"source": "analyst", "question": "Confirm the vendor?"},
    )
    assert check.source == "agent_request"
    assert "Confirm the vendor?" in check.question


def test_budget_exhaustion(settings):
    engine = PolicyEngine(settings)
    check = engine.evaluate(task="write a poem", confidence=0.9, iterations=99, max_iterations=8)
    assert check.needs_escalation
    assert check.source == "budget"


def test_already_reviewed_suppresses_repeat_escalation(settings):
    """After a human engages, the same policy trigger must not loop forever."""
    engine = PolicyEngine(settings)
    check = engine.evaluate(task="wire transfer money", confidence=0.3, already_reviewed=True)
    assert not check.needs_escalation


def test_apply_decision_to_state():
    update = EscalationManager.apply_decision_to_state(
        {"decision": "reject", "comment": "not authorized"}
    )
    assert "REJECTED" in update["human_feedback_text"]
    assert update["escalation_requested"] is None
    assert update["budget_exhausted"] is False

    update = EscalationManager.apply_decision_to_state({"decision": "modify", "modified_answer": "do X"})
    assert "MODIFIED" in update["human_feedback_text"]
    assert "do X" in update["human_feedback_text"]
