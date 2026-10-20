import { useCallback, useEffect, useState } from "react";
import {
  api,
  Escalation,
  MemoryHit,
  Run,
  RunDetail,
} from "./api";

const SAMPLE_TASK =
  "Research the impact of EU AI regulation on small startups and draft a concise briefing.";

const STATUS_ICON: Record<string, string> = {
  completed: "✅",
  awaiting_human: "⏸",
  failed: "❌",
  running: "🏃",
  queued: "⏳",
};

export default function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [escalations, setEscalations] = useState<Escalation[]>([]);
  const [task, setTask] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<string>("checking…");
  const [memQuery, setMemQuery] = useState("");
  const [memHits, setMemHits] = useState<MemoryHit[]>([]);
  const [comment, setComment] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [runList, pending, h] = await Promise.all([
        api.runs(),
        api.escalations("pending"),
        api.health(),
      ]);
      setRuns(runList);
      setEscalations(pending);
      setHealth(
        `${h.checks.llm.provider} · ${h.checks.llm.model} · queue ${h.queue_mode}`
      );
      if (selectedId) setDetail(await api.run(selectedId));
    } catch {
      setHealth("API unreachable");
    }
  }, [selectedId]);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 2500);
    return () => clearInterval(timer);
  }, [refresh]);

  async function submit() {
    if (!task.trim() || busy) return;
    setBusy(true);
    try {
      const run = await api.createRun(task.trim(), "async");
      setTask("");
      setSelectedId(run.id);
      await refresh();
    } finally {
      setBusy(false);
    }
  }

  async function resolve(id: string, decision: string) {
    await api.resolve(id, decision, comment);
    setComment("");
    await refresh();
  }

  async function search() {
    if (!memQuery.trim()) return;
    const res = await api.memorySearch(memQuery.trim());
    setMemHits(res.hits);
  }

  return (
    <div className="app">
      <header>
        <h1>⚛️ Agent Orchestration — React Console</h1>
        <span className="badge">{health}</span>
      </header>

      <main>
        <section className="panel">
          <h2>Delegate a task</h2>
          <textarea
            value={task}
            placeholder="What should the agent team do?"
            onChange={(e) => setTask(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && submit()}
          />
          <div className="row">
            <button className="btn" onClick={submit} disabled={busy}>
              ▶ Run agent team
            </button>
            <button className="btn ghost" onClick={() => setTask(SAMPLE_TASK)}>
              Sample
            </button>
          </div>

          <h2 style={{ marginTop: 20 }}>Recent runs</h2>
          {runs.map((run) => (
            <div
              key={run.id}
              className={`run-item ${run.id === selectedId ? "active" : ""}`}
              onClick={() => setSelectedId(run.id)}
            >
              <div className="task">{run.task}</div>
              <div className="row">
                <span className={`chip ${run.status}`}>
                  {STATUS_ICON[run.status] ?? "•"} {run.status.replace("_", " ")}
                </span>
                <span className="chip">{new Date(run.created_at).toLocaleTimeString()}</span>
              </div>
            </div>
          ))}
        </section>

        <section className="panel">
          <h2>Run detail</h2>
          {!detail && <p className="muted">Select a run.</p>}
          {detail && (
            <>
              <div className="row" style={{ marginBottom: 8 }}>
                <span className={`chip ${detail.status}`}>{detail.status}</span>
                <span className="chip">{detail.id.slice(0, 8)}</span>
                {(detail.output_data.route_history ?? []).length > 0 && (
                  <span className="chip">
                    {(detail.output_data.route_history ?? []).join(" → ")}
                  </span>
                )}
              </div>
              <p className="muted">{detail.task}</p>

              <h2>Supervisor log</h2>
              <div className="timeline" style={{ maxHeight: 110 }}>
                {(detail.output_data.supervisor_notes ?? []).map((note, i) => (
                  <div className="event" key={i}>
                    <span className="p">{note}</span>
                  </div>
                ))}
              </div>

              <h2>Activity timeline</h2>
              <div className="timeline">
                {detail.events.slice(-50).map((ev) => (
                  <div className="event" key={ev.id}>
                    <span className="t">
                      {new Date(ev.created_at).toLocaleTimeString()}
                    </span>
                    <span className="a">{ev.type}</span>
                    <span className="p">
                      {ev.actor} {JSON.stringify(ev.payload).slice(0, 110)}
                    </span>
                  </div>
                ))}
              </div>

              <h2>Final answer</h2>
              {detail.final_answer ? (
                <div className="answer">{detail.final_answer}</div>
              ) : (
                <p className="muted">
                  {detail.status === "awaiting_human"
                    ? "⏸ Waiting for a human decision…"
                    : "Working…"}
                </p>
              )}
            </>
          )}
        </section>

        <section>
          <div className="panel" style={{ marginBottom: 16 }}>
            <h2>🛑 Human-in-the-loop</h2>
            {escalations.length === 0 && (
              <p className="muted">No pending escalations 🎉</p>
            )}
            {escalations.map((esc) => (
              <div className="escalation" key={esc.id}>
                <h3>
                  {esc.kind === "final_review"
                    ? "📝 Final answer review"
                    : "🛑 Decision needed"}
                </h3>
                <p>
                  <b>{esc.question}</b>
                </p>
                <p>{esc.reason}</p>
                {esc.context.draft && (
                  <pre>{esc.context.draft.slice(0, 500)}</pre>
                )}
                <input
                  placeholder="Guidance / comment (optional)"
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                />
                <div className="row">
                  <button className="btn ok" onClick={() => resolve(esc.id, "approve")}>
                    ✔ Approve
                  </button>
                  <button
                    className="btn danger"
                    onClick={() => resolve(esc.id, "reject")}
                  >
                    ✖ Reject
                  </button>
                  <button className="btn warn" onClick={() => resolve(esc.id, "modify")}>
                    ✎ Modify
                  </button>
                </div>
              </div>
            ))}
          </div>

          <div className="panel">
            <h2>🧠 Long-term memory</h2>
            <input
              placeholder="Search semantic memory…"
              value={memQuery}
              onChange={(e) => setMemQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && search()}
            />
            <div className="row">
              <button className="btn ghost" onClick={search}>
                Search
              </button>
            </div>
            {memHits.map((hit) => (
              <div className="mem-hit" key={hit.id}>
                <b>{String(hit.metadata.kind ?? "memory")}</b> · score {hit.score}
                <br />
                {hit.text.slice(0, 220)}
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
