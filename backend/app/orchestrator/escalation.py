"""Human-in-the-loop escalation: policy engine + manager.

Escalations are raised when:

1. The supervisor explicitly routes to ``escalate`` (judgment call / ambiguity);
2. A worker calls the ``request_human`` tool (permission / policy boundary);
3. A policy rule matches the task or plan (risky or irreversible actions);
4. Supervisor confidence drops below the configured threshold;
5. The iteration budget is exhausted mid-plan.

Every escalation becomes a durable :class:`~app.memory.models.Escalation`
record *and* a LangGraph interrupt, so a human can approve / reject / modify
from the API or UI and the run resumes with that decision folded in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.memory.models import Escalation
from app.memory.store import RelationalStore


@dataclass
class PolicyRule:
    name: str
    description: str
    pattern: re.Pattern[str]
    applies_to: str = "task"  # task | plan | output


def _rule(name: str, description: str, pattern: str) -> PolicyRule:
    return PolicyRule(name=name, description=description, pattern=re.compile(pattern, re.I))


DEFAULT_RULES: list[PolicyRule] = [
    _rule(
        "financial_transaction",
        "Money movement or purchases require human approval",
        r"\b(wire\s*transfer|send\s+money|payment\s+of|purchase\s+order|refund\s+of|pay\s+an?\s+invoice|bitcoin\s+transfer)\b",
    ),
    _rule(
        "data_deletion",
        "Destructive data operations require human approval",
        r"\b(drop\s+table|truncate|delete\s+(all|everything|production)|purge|destroy\s+data|shred)\b",
    ),
    _rule(
        "pii_exposure",
        "Handling of personal identifiers requires human approval",
        r"\b(social\s+security|ssn|credit\s+card\s+number|passport\s+number|bank\s+account\s+number)\b",
    ),
    _rule(
        "credential_exposure",
        "Sharing secrets or credentials requires human approval",
        r"\b(password|api\s*key|private\s+key|secret)\b[\s\S]{0,40}\b(share|send|post|expose|publish|leak)\b",
    ),
    _rule(
        "irreversible_action",
        "Irreversible actions require human approval",
        r"\b(irreversible|permanent(ly)?\s+delete|cancel\s+all|revoke\s+all|wipe)\b",
    ),
    _rule(
        "external_communication",
        "Sending external communications requires human approval",
        r"\b(send\s+(an?\s+)?(email|message|tweet)|post\s+(publicly|online)|publish)\b",
    ),
]


@dataclass
class EscalationCheck:
    needs_escalation: bool
    reason: str = ""
    question: str = ""
    source: str = "policy"
    matched_rules: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "needs_escalation": self.needs_escalation,
            "reason": self.reason,
            "question": self.question,
            "source": self.source,
            "matched_rules": self.matched_rules,
        }


class PolicyEngine:
    """Declarative guardrails evaluated on every supervisor turn."""

    def __init__(self, settings: Settings, rules: list[PolicyRule] | None = None):
        self.settings = settings
        self.rules = rules if rules is not None else list(DEFAULT_RULES)

    def match_rules(self, text: str) -> list[PolicyRule]:
        return [r for r in self.rules if r.pattern.search(text or "")]

    def evaluate(
        self,
        *,
        task: str = "",
        plan: str = "",
        confidence: float = 1.0,
        explicit: bool = False,
        explicit_question: str = "",
        tool_request: dict | None = None,
        iterations: int = 0,
        max_iterations: int | None = None,
        already_reviewed: bool = False,
    ) -> EscalationCheck:
        matched = self.match_rules(f"{task}\n{plan}")
        # Once a human has engaged (approve/modify), the run carries a standing
        # mandate: don't re-escalate for the same policy/uncertainty triggers,
        # otherwise a single risky task would loop forever. Explicit supervisor
        # escalations and fresh agent tool requests still interrupt.
        if tool_request:
            return EscalationCheck(
                needs_escalation=True,
                reason=f"Agent '{tool_request.get('source')}' requested human input.",
                question=tool_request.get("question", "Human input required."),
                source="agent_request",
            )
        if matched and not already_reviewed:
            names = [r.name for r in matched]
            return EscalationCheck(
                needs_escalation=True,
                reason=f"Policy rules triggered: {', '.join(names)} — {matched[0].description}.",
                question=f"The planned action touches '{matched[0].name}' ({matched[0].description}). Proceed?",
                source="policy",
                matched_rules=names,
            )
        budget = max_iterations or self.settings.max_supervisor_iterations
        if iterations >= budget:
            return EscalationCheck(
                needs_escalation=True,
                reason=f"Iteration budget exhausted ({iterations}/{budget}).",
                question="The agents hit their iteration budget. Approve the best plan so far, or redirect them?",
                source="budget",
            )
        if explicit:
            return EscalationCheck(
                needs_escalation=True,
                reason="Supervisor requested human judgement.",
                question=explicit_question or "The supervisor needs a human decision to continue.",
                source="supervisor",
            )
        if (
            not already_reviewed
            and self.settings.escalation_on_low_confidence
            and confidence < self.settings.min_confidence
        ):
            return EscalationCheck(
                needs_escalation=True,
                reason=f"Supervisor confidence {confidence:.2f} below threshold {self.settings.min_confidence:.2f}.",
                question="The supervisor is not confident in the current plan. How should the team proceed?",
                source="low_confidence",
            )
        return EscalationCheck(needs_escalation=False)


class EscalationManager:
    """Persists escalations and turns human decisions into graph resume values."""

    def __init__(self, store: RelationalStore, settings: Settings):
        self.store = store
        self.settings = settings

    def create(
        self,
        run_id: str,
        kind: str,
        reason: str,
        question: str,
        context: dict | None = None,
    ) -> Escalation:
        return self.store.create_escalation(
            run_id=run_id, kind=kind, reason=reason, question=question, context=context or {}
        )

    def resolve(self, escalation_id: str, decision: dict) -> Escalation | None:
        """Persist a human decision. ``decision`` keys: decision, comment, modified_answer?"""
        verdict = str(decision.get("decision", "approve")).lower()
        status = {"approve": "approved", "reject": "rejected", "modify": "modified"}.get(verdict, "approved")
        return self.store.resolve_escalation(escalation_id, status=status, decision=decision)

    @staticmethod
    def apply_decision_to_state(decision: dict) -> dict[str, Any]:
        """Translate a human decision into state updates after interrupt resume."""
        verdict = str(decision.get("decision", "approve")).lower()
        comment = decision.get("comment") or ""
        if verdict == "approve":
            feedback = f"Human APPROVED the escalation.{(' Comment: ' + comment) if comment else ''}"
        elif verdict == "modify":
            detail = decision.get("modified_answer") or comment or "proceed with adjustments"
            feedback = f"Human MODIFIED the plan. Guidance: {detail}"
        else:
            feedback = f"Human REJECTED the proposed action.{(' Comment: ' + comment) if comment else ''} Revise the plan accordingly."
        return {
            "human_decision": decision,
            "human_feedback_text": feedback,
            "escalation_requested": None,
            "escalation": None,
            "status": "running",
            # Human intervention renews the run's mandate: give the supervisor
            # a fresh budget unless the human explicitly said to stop.
            "budget_exhausted": False,
        }
