"""Command-line entry point.

Examples
--------
    python -m app.cli run "Research X and draft a briefing"
    python -m app.cli demo
    python -m app.cli memory-search "invoice payment terms"
    python -m app.cli serve
"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_run(args: argparse.Namespace) -> int:
    from app.runtime import RunService

    service = RunService()
    result = service.submit_run(args.task, mode="sync")
    if result.status == "awaiting_human":
        pending = service.store.get_pending_escalation(result.id)
        print(json.dumps({"status": "awaiting_human", "run_id": result.id,
                          "escalation": {
                              "id": pending.id if pending else None,
                              "question": pending.question if pending else None,
                              "reason": pending.reason if pending else None,
                          }}, indent=2))
        print(f"\nResolve with: python -m app.cli resume {result.id} approve", file=sys.stderr)
        return 2
    print(result.final_answer or "(no answer)")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    from app.runtime import RunService

    service = RunService()
    result = service.resume_run(
        args.run_id,
        {"decision": args.decision, "comment": args.comment or ""},
    )
    if result.get("status") == "awaiting_human":
        print(json.dumps(result, indent=2))
        return 2
    print(result.get("final_answer") or "(no answer)")
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    from app.config import get_settings
    from app.demo import run_demo

    settings = get_settings()
    settings.llm_provider = "scripted"  # demo is fully offline
    run_demo()
    return 0


def cmd_memory_search(args: argparse.Namespace) -> int:
    from app.runtime import RunService

    service = RunService()
    hits = service.vector.search(args.query, k=args.k)
    for hit in hits:
        print(f"({hit.score:.3f}) {hit.text}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from app.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "app.api.app:create_app",
        factory=True,
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-orch", description="Agent Orchestration System CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Execute a task with the agent team")
    p_run.add_argument("task")
    p_run.set_defaults(func=cmd_run)

    p_resume = sub.add_parser("resume", help="Resolve an escalated run")
    p_resume.add_argument("run_id")
    p_resume.add_argument("decision", choices=["approve", "reject", "modify"])
    p_resume.add_argument("--comment", default="")
    p_resume.set_defaults(func=cmd_resume)

    p_demo = sub.add_parser("demo", help="Run the scripted offline demo (no API keys)")
    p_demo.set_defaults(func=cmd_demo)

    p_mem = sub.add_parser("memory-search", help="Search long-term semantic memory")
    p_mem.add_argument("query")
    p_mem.add_argument("-k", type=int, default=5)
    p_mem.set_defaults(func=cmd_memory_search)

    p_serve = sub.add_parser("serve", help="Start the API + web console")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
