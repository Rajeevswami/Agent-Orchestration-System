"""Persistent memory tests: relational store + ChromaDB vector memory."""

from __future__ import annotations


def test_conversation_and_run_lifecycle(store):
    conv = store.create_conversation(title="Test chat")
    run = store.create_run(conv.id, task="Do the thing", provider="scripted")

    assert run.status == "queued"
    store.update_run(run.id, status="running")
    store.update_run(run.id, status="completed", final_answer="done", output_data={"a": 1})
    store.update_run(run.id, output_data={"b": 2})

    fetched = store.get_run(run.id)
    assert fetched.status == "completed"
    assert fetched.final_answer == "done"
    assert fetched.output_data == {"a": 1, "b": 2}  # merged


def test_message_history_persists(store):
    conv = store.create_conversation()
    run = store.create_run(conv.id, task="hello")
    store.add_messages(
        [
            dict(run_id=run.id, conversation_id=conv.id, role="user", content="hello", seq=0),
            dict(run_id=run.id, conversation_id=conv.id, role="agent", name="researcher", content="hi", seq=1),
        ]
    )
    msgs = store.get_messages(conversation_id=conv.id)
    assert [m.role for m in msgs] == ["user", "agent"]
    assert msgs[1].name == "researcher"


def test_event_feed_ordered(store):
    conv = store.create_conversation()
    run = store.create_run(conv.id, task="x")
    store.add_event(run.id, "worker.started", actor="researcher", payload={"k": 1})
    store.add_event(run.id, "worker.completed", actor="researcher", payload={"k": 2})
    events = store.list_events(run.id)
    assert [e.type for e in events] == ["worker.started", "worker.completed"]
    assert store.list_events(run.id, since_id=events[0].id)[0].type == "worker.completed"


def test_escalation_create_is_idempotent_and_resolvable(store):
    conv = store.create_conversation()
    run = store.create_run(conv.id, task="wire money")
    first = store.create_escalation(run.id, kind="decision", reason="policy", question="Proceed?")
    second = store.create_escalation(run.id, kind="decision", reason="policy", question="Proceed again?")
    assert first.id == second.id  # one pending at a time

    resolved = store.resolve_escalation(first.id, status="approved", decision={"decision": "approve"})
    assert resolved.status == "approved"
    assert store.get_pending_escalation(run.id) is None


def test_vector_memory_search(vector):
    vector.reset()
    vector.add("Revenue grew 12% quarter over quarter", metadata={"kind": "semantic"})
    vector.add("The team prefers async standups on Mondays", metadata={"kind": "profile"})
    vector.add("Incident 42 was caused by a bad nginx config", metadata={"kind": "episodic"})

    hits = vector.search("revenue growth")
    assert hits and "Revenue" in hits[0].text

    hits = vector.search("what caused incident 42")
    assert hits and "nginx" in hits[0].text

    assert vector.count() == 3


def test_vector_memory_filters_by_metadata(vector):
    vector.reset()
    vector.add("User likes concise answers", metadata={"kind": "profile"})
    vector.add("Deploy happened at noon", metadata={"kind": "episodic"})
    hits = vector.search("answers", where={"kind": "profile"})
    assert all(h.metadata.get("kind") == "profile" for h in hits)
