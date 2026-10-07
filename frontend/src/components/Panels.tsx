// Side-panel widgets: KPI tiles, sparklines, the inspector and the comparison view.

import { useState } from "react";
import { api, type CompareResult, type Meta, type ScenarioRef } from "../lib/api";
import { remainingRoute, type Selection, type SessionController, type Snapshot } from "../lib/session";

export function fmt(v: unknown, digits = 1): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "–";
  return v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

export function clock(t: number): string {
  const s = Math.floor(t);
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

export function Sparkline({ values, color = "#38bdf8" }: { values: number[]; color?: string }) {
  const pts = values.filter((v) => Number.isFinite(v));
  if (pts.length < 2) return <div className="h-10 text-xs text-slate-500">collecting…</div>;
  const lo = Math.min(...pts), hi = Math.max(...pts);
  const span = hi - lo || 1;
  const w = 220, h = 40;
  const step = w / (values.length - 1);
  const d = values
    .map((v, i) => (Number.isFinite(v) ? `${i * step},${h - ((v - lo) / span) * (h - 4) - 2}` : null))
    .filter(Boolean)
    .join(" L");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-10 w-full" preserveAspectRatio="none" aria-hidden>
      <path d={`M${d}`} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

const KPIS: { key: string; label: string; unit: string; color: string; digits?: number }[] = [
  { key: "active", label: "Vehicles on network", unit: "", color: "#60a5fa", digits: 0 },
  { key: "speed_mean", label: "Mean speed", unit: "m/s", color: "#34d399" },
  { key: "queue_mean", label: "Network queue", unit: "veh", color: "#f87171" },
  { key: "throughput_vph", label: "Throughput", unit: "veh/h", color: "#fbbf24", digits: 0 },
];

export function MetricsPanel({ snap }: { snap: Snapshot }) {
  return (
    <div className="space-y-3">
      {KPIS.map((k) => (
        <div key={k.key} className="rounded-lg bg-slate-900/70 p-2">
          <div className="flex items-baseline justify-between">
            <span className="text-xs text-slate-400">{k.label}</span>
            <span className="font-mono text-sm text-slate-100">
              {fmt(snap.metrics[k.key], k.digits ?? 1)} <span className="text-slate-500">{k.unit}</span>
            </span>
          </div>
          <Sparkline values={snap.series[k.key] ?? []} color={k.color} />
        </div>
      ))}
      <p className="text-[11px] leading-snug text-slate-500">
        10-second samples computed by the engine's metrics manager (queue = time-mean of the
        network's stop-line queues; throughput = arrivals per hour).
      </p>
    </div>
  );
}

export function Inspector({
  selection,
  controller,
}: {
  selection: Selection | null;
  controller: SessionController;
}) {
  if (!selection) {
    return <p className="text-sm text-slate-400">Click a vehicle or an intersection on the map.</p>;
  }
  if (!selection.available) {
    return <p className="text-sm text-slate-400">{selection.id} has left the network.</p>;
  }
  const d = selection.detail ?? {};
  if (selection.kind === "vehicle") {
    const rows: [string, string][] = [
      ["Type", String(d.type)],
      ["Road / lane", `${d.road ?? "–"} / ${d.lane ?? "–"}`],
      ["Speed", `${fmt(d.speed)} m/s (desired ${fmt(d.desired_speed)})`],
      ["Acceleration", `${fmt(d.acceleration, 2)} m/s²`],
      ["Waiting time", `${fmt(d.waiting_time)} s`],
      ["Stops", fmt(d.stops, 0)],
      ["Distance", `${fmt(d.distance, 0)} m`],
      ["Destination", String(d.destination ?? "–")],
    ];
    const route = remainingRoute(selection);
    if (route) rows.push(["Route ahead", route.join(" → ")]);
    return (
      <div>
        <h3 className="mb-2 font-mono text-sm text-sky-300">{selection.id}</h3>
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
          {rows.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-slate-400">{k}</dt>
              <dd className="text-slate-100">{v}</dd>
            </div>
          ))}
        </dl>
      </div>
    );
  }
  const phases = (d.phases as { index: number; id: string; duration: number }[]) ?? [];
  const states = (d.movement_states as Record<string, string>) ?? {};
  return (
    <div>
      <h3 className="mb-1 font-mono text-sm text-sky-300">Intersection {selection.id}</h3>
      <p className="mb-2 text-xs text-slate-400">
        {String(d.controller)} · phase <b className="text-slate-100">{String(d.phase_id)}</b> ·{" "}
        {String(d.stage)}
        {d.remaining != null && ` · ${fmt(d.remaining)} s left`}
        {d.held ? " · HELD" : ""}
        {d.preempting ? <b className="text-rose-400"> · EMERGENCY PREEMPTION</b> : null}
      </p>
      <div className="mb-2 flex flex-wrap gap-1">
        {phases.map((p) => (
          <button
            key={p.index}
            onClick={() => controller.send({ type: "hold_phase", intersection: selection.id, phase: p.index })}
            className={`rounded px-2 py-1 text-xs ${
              p.index === d.phase_index ? "bg-emerald-700 text-white" : "bg-slate-800 hover:bg-slate-700"
            }`}
            title="Hold this phase (manual override)"
          >
            Hold {p.id}
          </button>
        ))}
        {Boolean(d.held) && (
          <button
            onClick={() => controller.send({ type: "release_phase", intersection: selection.id })}
            className="rounded bg-amber-700 px-2 py-1 text-xs hover:bg-amber-600"
          >
            Release
          </button>
        )}
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-[11px]">
        {Object.entries(states).map(([m, s]) => (
          <div key={m} className="flex justify-between">
            <span className="text-slate-400">{m}</span>
            <span
              className={
                s === "G" ? "text-emerald-400" : s === "g" ? "text-emerald-200" : s === "y" ? "text-yellow-300" : "text-red-400"
              }
            >
              {s}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

const LABELS: Record<string, string> = {
  "travel_time.mean": "Mean travel time (s)",
  "delay.mean": "Mean delay (s)",
  "waiting_time_mean": "Mean waiting time (s)",
  "stops_mean": "Stops per vehicle",
  "throughput_vph": "Throughput (veh/h)",
  "queue.mean_total_veh": "Mean network queue (veh)",
  "vehicles.arrived": "Vehicles arrived",
};

export function ComparePanel({ meta, scenarios }: { meta: Meta; scenarios: ScenarioRef[] }) {
  const [scenario, setScenario] = useState(scenarios.find((s) => s.name === "grid_3x3")?.name ?? scenarios[0]?.name ?? "");
  const [chosen, setChosen] = useState<string[]>(["fixed_time", "actuated", "max_pressure"].filter((c) => meta.controllers.includes(c)));
  const [seeds, setSeeds] = useState(3);
  const [duration, setDuration] = useState(1800);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.compare({ scenario, controllers: chosen, seeds, duration }));
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3 text-sm">
      <label className="block">
        <span className="text-xs text-slate-400">Scenario</span>
        <select className="mt-1 w-full rounded bg-slate-800 p-1" value={scenario} onChange={(e) => setScenario(e.target.value)}>
          {scenarios.filter((s) => s.source === "bundled").map((s) => (
            <option key={s.name}>{s.name}</option>
          ))}
        </select>
      </label>
      <fieldset>
        <legend className="text-xs text-slate-400">Controllers (first = baseline)</legend>
        <div className="mt-1 flex flex-wrap gap-2">
          {meta.controllers.map((c) => (
            <label key={c} className="flex items-center gap-1 text-xs">
              <input
                type="checkbox"
                checked={chosen.includes(c)}
                onChange={(e) => setChosen(e.target.checked ? [...chosen, c] : chosen.filter((x) => x !== c))}
              />
              {c}
            </label>
          ))}
        </div>
      </fieldset>
      <div className="flex gap-2">
        <label className="flex-1">
          <span className="text-xs text-slate-400">Seeds</span>
          <input type="number" min={1} max={10} value={seeds} onChange={(e) => setSeeds(+e.target.value)} className="mt-1 w-full rounded bg-slate-800 p-1" />
        </label>
        <label className="flex-1">
          <span className="text-xs text-slate-400">Duration (s)</span>
          <input type="number" min={60} step={60} value={duration} onChange={(e) => setDuration(+e.target.value)} className="mt-1 w-full rounded bg-slate-800 p-1" />
        </label>
      </div>
      <button
        disabled={busy || chosen.length < 2}
        onClick={run}
        className="w-full rounded bg-sky-600 py-1.5 font-medium hover:bg-sky-500 disabled:opacity-50"
      >
        {busy ? `Running ${chosen.length * seeds} simulations…` : "Run comparison"}
      </button>
      {error && <pre className="whitespace-pre-wrap rounded bg-red-950 p-2 text-xs text-red-200">{error}</pre>}
      {result && <CompareTable result={result} />}
    </div>
  );
}

function CompareTable({ result }: { result: CompareResult }) {
  const others = result.controllers.filter((c) => c !== result.baseline);
  return (
    <div className="space-y-3">
      <p className="text-xs text-slate-400">
        {result.scenario}, {result.seeds.length} seeds × {result.duration ?? "default"} s. Mean [95% CI]; Δ is
        paired against <b>{result.baseline}</b> (★ = CI excludes 0). Same seeds ⇒ identical demand.
      </p>
      {result.metrics.map((row) => {
        const means = result.controllers.map((c) => row.by_controller[c]?.mean ?? NaN);
        const max = Math.max(...means.filter(Number.isFinite), 1e-9);
        return (
          <div key={row.metric} className="rounded-lg bg-slate-900/70 p-2">
            <div className="mb-1 text-xs font-medium text-slate-300">{LABELS[row.metric] ?? row.metric}</div>
            {result.controllers.map((c) => {
              const st = row.by_controller[c]!;
              const delta = row.vs_baseline[c];
              const good =
                delta && delta.significant && delta.delta != null && row.direction
                  ? (row.direction === "min") === delta.delta < 0
                  : null;
              return (
                <div key={c} className="mb-0.5 flex items-center gap-2 text-xs">
                  <span className="w-24 truncate text-slate-400">{c}</span>
                  <div className="h-3 flex-1 rounded bg-slate-800">
                    <div className="h-3 rounded bg-sky-500" style={{ width: `${(100 * (st.mean ?? 0)) / max}%` }} />
                  </div>
                  <span className="w-16 text-right font-mono">{fmt(st.mean)}</span>
                  <span
                    className={`w-20 text-right font-mono ${good == null ? "text-slate-500" : good ? "text-emerald-400" : "text-red-400"}`}
                  >
                    {delta?.delta_pct != null ? `${delta.delta_pct > 0 ? "+" : ""}${fmt(delta.delta_pct)}%${delta.significant ? "★" : ""}` : c === result.baseline ? "baseline" : "–"}
                  </span>
                </div>
              );
            })}
          </div>
        );
      })}
      {others.length === 0 && <p className="text-xs text-slate-500">Pick at least two controllers.</p>}
    </div>
  );
}
