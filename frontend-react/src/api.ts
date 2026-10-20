export interface Run {
  id: string;
  conversation_id: string;
  task: string;
  status: string;
  provider: string;
  final_answer: string | null;
  error: string | null;
  created_at: string;
  output_data: { route_history?: string[]; supervisor_notes?: string[] };
}

export interface RunEvent {
  id: number;
  type: string;
  actor: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface Escalation {
  id: string;
  run_id: string;
  kind: string;
  status: string;
  reason: string;
  question: string;
  context: { task?: string; draft?: string };
}

export interface RunDetail extends Run {
  events: RunEvent[];
  pending_escalation: Escalation | null;
}

export interface MemoryHit {
  id: string;
  text: string;
  metadata: Record<string, unknown>;
  score: number;
}

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ version: string; queue_mode: string; checks: any }>("/health"),
  runs: (limit = 25) => request<Run[]>(`/runs?limit=${limit}`),
  run: (id: string) => request<RunDetail>(`/runs/${id}`),
  createRun: (task: string, mode: "sync" | "async" = "async") =>
    request<Run>("/runs", { method: "POST", body: JSON.stringify({ task, mode }) }),
  escalations: (status = "pending") =>
    request<Escalation[]>(`/escalations?status=${status}`),
  resolve: (id: string, decision: string, comment = "") =>
    request(`/escalations/${id}/resolve`, {
      method: "POST",
      body: JSON.stringify({ decision, comment }),
    }),
  memorySearch: (q: string, k = 6) =>
    request<{ hits: MemoryHit[] }>(`/memory/search?q=${encodeURIComponent(q)}&k=${k}`),
};
