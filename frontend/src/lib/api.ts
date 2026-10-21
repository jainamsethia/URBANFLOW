// REST client and the server's JSON shapes.

export type Point = [number, number];

export interface Geometry {
  version: number;
  geometry_crc: number;
  origin: Point;
  bbox: [number, number, number, number];
  links: { id: string[]; kind: ("lane" | "connector")[]; owner: string[]; lane_width: number[]; path: Point[][] };
  movements: { id: string[]; intersection: string[]; from_road: string[]; to_road: string[]; turn: string[]; links: number[][] };
  signal_heads: { movement: number[]; position: Point[] };
  roads: { id: string[]; surfaces: Point[][] };
  intersections: { id: string[]; kind: string[]; point: Point[]; radius: number[]; polygons: Point[][] };
  lane_markings: { road: number[]; dashed: boolean[]; path: Point[][] };
  stop_lines: { link: number[]; path: Point[][] };
  vehicle_types: { id: string; vclass: string; length: number; width: number; color: string | null }[];
  bus_stops?: { id: string[]; position: Point[] };
}

export interface LaneMetricInfo {
  code: number;
  name: string;
  label: string;
  unit: string;
  domain: [number, number] | null;
}

export interface Meta {
  version: string;
  project: string;
  controllers: string[];
  generators: { name: string; description: string }[];
  lane_metrics: LaneMetricInfo[];
  auth_required: boolean;
}

export interface ScenarioRef {
  name: string;
  source: "bundled" | "workspace";
}

export interface SessionInfo {
  id: string;
  label: string;
  scenario: string;
  state: string;
  step: number;
  time: number;
  end_time: number | null;
  dt: number;
  steps_per_second: number | null;
  vehicles: number;
  controllers: Record<string, string>;
  subscribers: number;
}

export interface CompareStat {
  mean: number | null;
  std: number | null;
  ci95: [number | null, number | null];
  values: (number | null)[];
}

export interface CompareResult {
  scenario: string;
  controllers: string[];
  baseline: string;
  seeds: number[];
  duration: number | null;
  metrics: {
    metric: string;
    direction: "min" | "max" | null;
    by_controller: Record<string, CompareStat>;
    vs_baseline: Record<string, { delta: number | null; delta_pct: number | null; ci95: [number | null, number | null]; n_pairs: number; significant: boolean }>;
  }[];
}

export type CompareRow = CompareResult["metrics"][number];

export interface ExperimentSummary {
  id: string;
  name: string;
  description: string;
  status: string;
  scenario: string | null;
  variants: string[];
  seeds: number[];
  baseline: string;
  progress: Record<string, number> & { total: number };
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface ExperimentRun {
  id: string;
  variant: string;
  seed: number;
  status: string;
  progress: number;
  wall_time_s: number | null;
  error: string | null;
  replay: boolean;
}

export interface ExperimentDoc extends ExperimentSummary {
  spec: { base_config: Record<string, unknown>; record_replay: boolean | string; [k: string]: unknown };
  runs: ExperimentRun[];
}

export interface ExperimentComparison {
  name: string;
  variants: string[];
  baseline: string;
  seeds: number[];
  status: string;
  missing: string[];
  metrics: CompareRow[];
}

export interface ExperimentSeries {
  variant: string;
  runs: number;
  time: number[];
  mean: (number | null)[];
  lo: (number | null)[];
  hi: (number | null)[];
}

const API = "/api/v1";

// A protected server prints a link ending in #token=...; keep the token in a same-site
// cookie (sent with every request, download and the live connection) and tidy the URL.
function keepToken(token: string) {
  document.cookie = `urbanflow_token=${encodeURIComponent(token)}; path=/; SameSite=Strict`;
}
const fromLink = new URLSearchParams(location.hash.slice(1)).get("token");
if (fromLink) {
  keepToken(fromLink);
  history.replaceState(null, "", location.pathname + location.search);
}
let asking = false;

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (res.status === 401 && !asking) {
    asking = true;
    const token = prompt("This UrbanFlow workbench is protected.\nPaste the access token (the part after #token= in the link it printed):");
    if (token?.trim()) {
      keepToken(token.trim());
      location.reload();
    }
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
      if (Array.isArray(body.issues) && body.issues.length && !String(detail).includes(body.issues[0].message)) {
        detail += "\n" + body.issues.map((i: { path: string; message: string }) => `${i.path}: ${i.message}`).join("\n");
      }
    } catch {
      /* not JSON */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const api = {
  meta: () => call<Meta>("/meta"),
  scenarios: () => call<ScenarioRef[]>("/scenarios"),
  sessions: () => call<SessionInfo[]>("/sessions"),
  replays: () => call<{ name: string; size_bytes: number; source: string }[]>("/replays"),
  createSession: (body: Record<string, unknown>) =>
    call<SessionInfo>("/sessions", { method: "POST", body: JSON.stringify(body) }),
  deleteSession: (id: string) => call<void>(`/sessions/${id}`, { method: "DELETE" }),
  geometry: (id: string) => call<Geometry>(`/sessions/${id}/geometry`),
  summary: (id: string) =>
    call<{ summary: Record<string, number | null>; history: Record<string, (number | null)[]> }>(`/sessions/${id}/summary`),
  compare: (body: Record<string, unknown>) =>
    call<CompareResult>("/compare", { method: "POST", body: JSON.stringify(body) }),
  ops: (scenario: unknown, ops: Record<string, unknown>[]) =>
    call<OpsResult>("/scenarios/ops", { method: "POST", body: JSON.stringify({ scenario, ops }) }),
  scenarioDoc: (source: string, name: string) => call<Record<string, unknown>>(`/scenarios/${source}/${encodeURIComponent(name)}`),
  importScenario: (body: Record<string, unknown>) =>
    call<ImportResult>("/scenarios/import", { method: "POST", body: JSON.stringify(body) }),
  deleteScenario: (name: string) => call<void>(`/scenarios/${encodeURIComponent(name)}`, { method: "DELETE" }),
  experiments: () => call<{ items: ExperimentSummary[]; total: number }>("/experiments?limit=200"),
  experiment: (id: string) => call<ExperimentDoc>(`/experiments/${id}`),
  createExperiment: (spec: Record<string, unknown>, run: boolean) =>
    call<ExperimentDoc>("/experiments", { method: "POST", body: JSON.stringify({ spec, run }) }),
  runExperiment: (id: string) => call<ExperimentDoc>(`/experiments/${id}/run`, { method: "POST", body: JSON.stringify({ resume: true }) }),
  cancelExperiment: (id: string) => call<ExperimentDoc>(`/experiments/${id}/cancel`, { method: "POST" }),
  deleteExperiment: (id: string) => call<void>(`/experiments/${id}?purge=true`, { method: "DELETE" }),
  experimentCompare: (id: string, baseline: string) =>
    call<ExperimentComparison>(`/experiments/${id}/compare?baseline=${encodeURIComponent(baseline)}`),
  experimentTimeseries: (id: string, column: string) =>
    call<{ column: string; series: ExperimentSeries[] }>(`/experiments/${id}/timeseries?column=${column}`),
};

export interface OpsResult {
  scenario: Record<string, unknown>;
  valid: boolean;
  issues: ImportIssue[];
  created: { op_index: number; kind: string; id: string }[];
}

export interface ImportIssue {
  path: string;
  message: string;
  code: string;
  severity: string;
}

export interface ImportResult {
  name: string;
  summary: Record<string, number>;
  warnings: ImportIssue[];
}

/** Download links (served as attachments). */
export const downloads = {
  table: (sessionId: string, table: "trips" | "timeseries" | "intersections") => `${API}/sessions/${sessionId}/export/${table}.csv`,
  sessionScenario: (sessionId: string) => `${API}/sessions/${sessionId}/scenario`,
  scenario: (source: string, name: string) => `${API}/scenarios/${source}/${encodeURIComponent(name)}`,
  replay: (name: string) => `${API}/replays/${encodeURIComponent(name)}/file`,
  experiment: (id: string, what: "runs" | "comparison", format: "csv" | "json") => `${API}/experiments/${id}/export?what=${what}&format=${format}`,
};

export function wsUrl(sessionId: string): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${API}/ws/sessions/${sessionId}`;
}
