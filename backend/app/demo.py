"""Offline demo scenario: the full orchestration loop with scripted LLMs.

Runs policy gate (wire transfer -> escalation -> human approval) -> supervisor
delegation -> tool use (web_search, calculator, memory) -> final answer ->
memory consolidation — with zero API keys and zero network access.

Used by ``python -m app.cli demo`` and as the golden path in the test suite.
"""

from __future__ import annotations

from typing import Any

from app.llm.scripted import register_script
from app.runtime import RunService

DEMO_TASK = (
    "We need to pay invoice #4821 for $2,400 to Northwind Supplies by wire transfer. "
    "Research their payment terms, calculate the total including the 2.5% processing "
    "fee, and draft the approval summary for finance."
)


def register_demo_script() -> None:
    """Install scripted models for every role (supervisor + workers)."""
    register_script(
        "supervisor",
        [
            # turn 1 — plan research, but the wire-transfer policy gate fires first
            '{"action": "researcher", "reasoning": "Need payment terms and invoice context before any math.", '
            '"instructions": "Find Northwind Supplies payment terms and any late-fee policy; check memory for prior invoices.", '
            '"confidence": 0.9}',
            # turn 2 — human approved the escalation; execute the research plan
            '{"action": "researcher", "reasoning": "Human approved the wire-transfer plan; gathering payment terms now.", '
            '"instructions": "Find Northwind Supplies payment terms and any late-fee policy; check memory for prior invoices.", '
            '"confidence": 0.9}',
            # turn 3 — delegate analysis
            '{"action": "analyst", "reasoning": "Research is in; now compute the fee-inclusive total and risk flags.", '
            '"instructions": "Compute the 2.5% processing fee on $2,400 and the total due. Cross-check anything suspicious.", '
            '"confidence": 0.85}',
            # turn 4 — delegate the write-up
            '{"action": "writer", "reasoning": "All numbers are in; draft the finance approval summary.", '
            '"instructions": "Draft the approval summary for finance with the totals and a recommendation.", '
            '"confidence": 0.9}',
            # turn 5 — done
            '{"action": "FINISH", "reasoning": "Writer delivered the complete approval summary.", "instructions": "", "confidence": 0.95}',
        ],
    )
    register_script(
        "researcher",
        [
            {
                "tool_calls": [
                    {"name": "retrieve_memory", "args": {"query": "Northwind Supplies invoice payment terms"}},
                    {"name": "web_search", "args": {"query": "Northwind Supplies payment terms net 30 late fee"}},
                ]
            },
            "Northwind Supplies operates on Net 30 payment terms with a 1.5% late fee after day 45. "
            "A prior invoice (#4610, March) was paid on day 27 with no fee. Memory shows finance policy "
            "requires written approval for any wire over $1,000.\n\n"
            "SUMMARY: Northwind terms are Net 30 (1.5% late fee after day 45). Prior invoice #4610 paid "
            "on time. Finance policy requires written approval for wires over $1,000 — this $2,400 "
            "payment needs that approval and the processing fee should be quantified before drafting.",
        ],
    )
    register_script(
        "analyst",
        [
            {
                "tool_calls": [
                    {"name": "calculator", "args": {"expression": "2400 * 0.025"}},
                    {"name": "calculator", "args": {"expression": "2400 + 60"}},
                ]
            },
            "The 2.5% processing fee on the $2,400 invoice is $60.00, so the total due is $2,460.00. "
            "Compared with the prior invoice paid at day 27, paying on day 30 avoids the 1.5% late fee ($36).\n\n"
            "SUMMARY: Fee $60.00 + invoice $2,400.00 = $2,460.00 total. Paying by day 30 avoids the $36 late "
            "fee. The payment is a wire transfer over $1,000, which is policy-sensitive and needs explicit "
            "human approval before any money moves.",
        ],
    )
    register_script(
        "writer",
        [
            "## Payment Approval Summary — Invoice #4821 (Northwind Supplies)\n\n"
            "**Amount:** $2,400.00 (invoice) + $60.00 (2.5% processing fee) = **$2,460.00 total**.\n"
            "**Terms:** Net 30; 1.5% late fee ($36) applies after day 45. Prior invoice #4610 was paid on day 27.\n"
            "**Policy:** Finance policy requires written approval for wires above $1,000 — a human has "
            "approved this escalation.\n\n"
            "**Recommendation:** Execute the wire transfer of $2,460.00 before the Net 30 due date and "
            "record the approval alongside invoice #4821.\n\n"
            "SUMMARY: Wire $2,460.00 ($2,400 + $60 fee) to Northwind Supplies for invoice #4821 before "
            "the Net 30 due date; human approval is on file and the late-fee risk ($36) is avoided.",
        ],
    )


def run_demo(service: RunService | None = None, verbose: bool = True) -> dict[str, Any]:
    """Execute the offline demo end-to-end, auto-resolving escalations."""
    register_demo_script()
    service = service or RunService()

    if verbose:
        print("\n\033[1m▶ Task for the agent team:\033[0m")
        print(f"  {DEMO_TASK}\n")

    run = service.create_run(DEMO_TASK)
    result = service.execute_run(run.id)

    if verbose:
        _print_flow(service, run.id)

    # The demo auto-approves escalations like a human operator would.
    while result.get("status") == "awaiting_human":
        if verbose:
            esc = result.get("escalation", {})
            print("\n\033[1m⏸ ESCALATION — human-in-the-loop\033[0m")
            print(f"  source:   {esc.get('source', '')}")
            print(f"  reason:   {esc.get('reason', '')}")
            print(f"  question: {esc.get('question', '')}")
            print("\033[1m  ✔ demo operator approves\033[0m\n")
        result = service.resume_run(
            run.id,
            {"decision": "approve", "comment": "Approved by demo operator — finance policy satisfied."},
        )
        if verbose:
            _print_flow(service, run.id)

    if verbose:
        print("\n\033[1m✔ FINAL ANSWER\033[0m")
        print(f"{result.get('final_answer', '(none)')}\n")
        run_obj = service.store.get_run(run.id)
        routes = " -> ".join((run_obj.output_data or {}).get("route_history", [])) if run_obj else ""
        print(f"  status:   {run_obj.status if run_obj else '?'}")
        print(f"  routes:   {routes}")
        hits = service.vector.search("invoice payment terms", k=3)
        top = hits[0].text[:80] if hits else "(none)"
        print(f"  memory:   {service.vector.count()} item(s), top hit: {top}…")
        events = service.store.list_events(run.id)
        print(f"  events:   {len(events)} timeline entries recorded")

    return result


def _print_flow(service: RunService, run_id: str) -> None:
    run = service.store.get_run(run_id)
    if run is None:
        return
    notes = (run.output_data or {}).get("supervisor_notes") or []
    if notes:
        print("\n\033[1m▶ Supervisor decisions:\033[0m")
        for note in notes:
            print(f"  {note}")
    events = service.store.list_events(run_id)
    for ev in events[-8:]:
        payload = str(ev.payload)
        print(f"  \033[36m{ev.type:22}\033[0m {ev.actor or '-':24} {payload[:90]}")
