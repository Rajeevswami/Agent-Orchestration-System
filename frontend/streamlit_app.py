"""Streamlit frontend for the Agent Orchestration System.

Run against the FastAPI backend:

    streamlit run frontend/streamlit_app.py

Set ``API_URL`` if the API is not on ``http://localhost:8000``.
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Agent Orchestration System",
    page_icon="🤖",
    layout="wide",
)


# ── API helpers ──────────────────────────────────────────────────────────────

def api(path: str, method: str = "GET", **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def health_ok() -> dict | None:
    return api("/api/health")


# ── Session state ────────────────────────────────────────────────────────────
for key, default in {
    "messages": [],          # chat transcript for the active conversation
    "run_ids": [],           # runs created in this session
    "selected_run": None,
}.items():
    st.session_state.setdefault(key, default)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Agent Orchestration")
    health = health_ok()
    if health:
        checks = health.get("checks", {})
        llm = checks.get("llm", {})
        st.success(f"API v{health.get('version')} · queue: {health.get('queue_mode')}")
        st.caption(f"LLM: **{llm.get('provider')}** · model `{llm.get('model')}`")
        for name, value in checks.items():
            if name != "llm":
                st.caption(f"{name}: {value}")
    else:
        st.error("API unreachable — start it with `python -m app.cli serve`")

    st.divider()
    require_approval = st.toggle(
        "Human review of final answers", value=False,
        help="Mirrors REQUIRE_FINAL_APPROVAL: every final answer gets an approval gate.",
    )
    st.caption("Escalations pause the run until you approve/reject below.")

    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.session_state.run_ids = []
        st.session_state.selected_run = None
        st.rerun()


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_chat, tab_runs, tab_escalations, tab_memory = st.tabs(
    ["💬 Chat", "🏃 Runs", "🛑 Escalations", "🧠 Memory"]
)

# ── Chat tab ─────────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Delegate a task to the agent team")
    st.caption(
        "The **supervisor** plans and delegates to **researcher / analyst / writer** "
        "agents. Risky or ambiguous work pauses for your approval."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("activity"):
                with st.expander("🔎 Agent activity"):
                    for line in msg["activity"]:
                        st.text(line)

    prompt = st.chat_input("What should the agent team do?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("🧑‍💼 Supervisor is planning…", expanded=False) as status:
                run = api(
                    "/api/runs",
                    method="POST",
                    json={
                        "task": prompt,
                        "mode": "sync",
                        "metadata": {"require_final_approval": require_approval},
                    },
                )
                if run is None:
                    status.update(label="Failed to start run", state="error")
                    st.stop()

                st.session_state.run_ids.append(run["id"])
                st.session_state.selected_run = run["id"]

                # Pull the activity timeline for the transcript.
                detail = api(f"/api/runs/{run['id']}") or {}
                activity = [
                    f"{e['type']}  {e['actor']}  {str(e['payload'])[:80]}"
                    for e in detail.get("events", [])
                ]

                if run.get("status") == "awaiting_human":
                    pending = detail.get("pending_escalation") or {}
                    status.update(label="⏸ Escalated to a human", state="warning")
                    st.warning(f"**Human input needed** — {pending.get('question', 'see Escalations tab')}")
                    answer = (
                        f"⏸ **Escalated:** {pending.get('reason', '')}\n\n"
                        f"> {pending.get('question', '')}\n\n"
                        "_Resolve it in the **Escalations** tab to continue the run._"
                    )
                else:
                    status.update(label="✅ Run completed", state="complete")
                    answer = run.get("final_answer") or "(no answer produced)"

                st.markdown(answer)
                with st.expander("🔎 Agent activity"):
                    for line in activity:
                        st.text(line)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "activity": activity}
            )

# ── Runs tab ─────────────────────────────────────────────────────────────────
with tab_runs:
    st.subheader("Run history")
    runs = api("/api/runs?limit=25") or []
    if not runs:
        st.info("No runs yet.")
    for run in runs:
        icon = {"completed": "✅", "awaiting_human": "⏸", "failed": "❌", "running": "🏃"}.get(run["status"], "⏳")
        with st.expander(f"{icon} {run['task'][:80]} — `{run['status']}`"):
            st.write(f"**Status:** {run['status']} · **Provider:** {run['provider']} · **ID:** `{run['id']}`")
            if run.get("final_answer"):
                st.markdown("**Final answer**")
                st.markdown(run["final_answer"])
            detail = api(f"/api/runs/{run['id']}") or {}
            routes = detail.get("output_data", {}).get("route_history") or []
            if routes:
                st.caption("Supervisor route: " + " → ".join(routes))
            events = detail.get("events", [])
            if events:
                st.markdown("**Activity timeline**")
                for e in events[-25:]:
                    st.text(f"{e['type']:22} {e['actor']:22} {str(e['payload'])[:90]}")

# ── Escalations tab ──────────────────────────────────────────────────────────
with tab_escalations:
    st.subheader("Human-in-the-loop queue")
    pending = api("/api/escalations?status=pending") or []
    if not pending:
        st.success("No pending escalations 🎉")
    for esc in pending:
        with st.container(border=True):
            kind = "📝 Final answer review" if esc["kind"] == "final_review" else "🛑 Decision needed"
            st.markdown(f"### {kind}")
            st.markdown(f"**{esc['question']}**")
            st.caption(esc["reason"])
            context = esc.get("context") or {}
            if context.get("draft"):
                st.markdown("**Draft answer**")
                st.code(context["draft"], language="markdown")
            if context.get("task"):
                st.caption(f"Task: {context['task']}")
            comment = st.text_input("Guidance / comment", key=f"c-{esc['id']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✔ Approve", key=f"a-{esc['id']}", type="primary"):
                    api(f"/api/escalations/{esc['id']}/resolve", method="POST",
                        json={"decision": "approve", "comment": comment})
                    st.rerun()
            with col2:
                if st.button("✖ Reject", key=f"r-{esc['id']}"):
                    api(f"/api/escalations/{esc['id']}/resolve", method="POST",
                        json={"decision": "reject", "comment": comment})
                    st.rerun()
            with col3:
                if st.button("✎ Modify", key=f"m-{esc['id']}"):
                    api(f"/api/escalations/{esc['id']}/resolve", method="POST",
                        json={"decision": "modify", "comment": comment,
                              "modified_answer": comment})
                    st.rerun()

    st.divider()
    st.subheader("Recent escalation history")
    resolved = [e for e in (api("/api/escalations?limit=25") or []) if e["status"] != "pending"]
    for esc in resolved[:10]:
        st.caption(
            f"`{esc['status']}` · {esc['question'][:90]} — "
            f"{(esc.get('decision') or {}).get('comment', '')[:60]}"
        )

# ── Memory tab ───────────────────────────────────────────────────────────────
with tab_memory:
    st.subheader("Long-term semantic memory (ChromaDB)")
    query = st.text_input("Search memory", placeholder="e.g. payment terms invoice")
    if query:
        hits = api(f"/api/memory/search?q={requests.utils.quote(query)}&k=8") or {}
        for hit in hits.get("hits", []):
            with st.container(border=True):
                meta = hit.get("metadata") or {}
                st.markdown(f"**{meta.get('kind', 'memory')}** · score `{hit['score']}`")
                st.write(hit["text"])
    st.divider()
    st.subheader("Recent memory records")
    records = api("/api/memory/records?limit=20") or []
    for rec in records:
        with st.expander(f"{rec['kind']} · {rec['content'][:70]}"):
            st.write(rec["content"])
            st.caption(str(rec.get("metadata", {})))
