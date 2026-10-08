// Side-panel widgets in plain language: traffic mood, stat tiles, the car and traffic-light
// cards, the controller race, and the technical views kept for grown-ups.

import { useState } from "react";
import { api, type CompareResult, type Geometry, type Meta, type ScenarioRef } from "../lib/api";
import { brain, brainChoices, clock, kmh, mood, movementSides, seconds, town, TOWN_ORDER, townSide, VEHICLE } from "../lib/friendly";
import { remainingRoute, type Selection, type SessionController, type Snapshot } from "../lib/session";

export function fmt(v: unknown, digits = 1): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "–";
  return v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

const card = "rounded-2xl bg-white p-4 shadow-sm ring-1 ring-slate-200";

// ---------------------------------------------------------------------------- traffic tab
export function MoodCard({ snap }: { snap: Snapshot }) {
  const m = mood(snap.metrics.active, snap.metrics.halting);
  return (
    <div className={`${card} flex items-center gap-3`} style={{ borderLeft: `6px solid ${m.color}` }}>
      <span className="text-5xl" aria-hidden>{m.face}</span>
      <div>
        <div className="text-xs font-medium uppercase tracking-wide text-slate-500">How is traffic?</div>
        <div className="text-xl font-semibold text-slate-900">{m.text}</div>
      </div>
    </div>
  );
}

function Sparkline({ values, times, format }: { values: number[]; times: number[]; format: (v: number) => string }) {
  const [hover, setHover] = useState<number | null>(null);
  const idx = values.map((v, i) => (Number.isFinite(v) ? i : -1)).filter((i) => i >= 0);
  if (idx.length < 2) return <div className="h-10 text-xs text-slate-400">Watching…</div>;
  const pts = idx.map((i) => values[i]!);
  const lo = Math.min(...pts), hi = Math.max(...pts);
  const span = hi - lo || 1;
  const w = 240, h = 40;
  const x = (i: number) => (i / (values.length - 1)) * w;
  const y = (v: number) => h - 3 - ((v - lo) / span) * (h - 6);
  const d = idx.map((i, k) => `${k ? "L" : "M"}${x(i)},${y(values[i]!)}`).join(" ");
  const last = idx[idx.length - 1]!;
  const shown = hover ?? last;
  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="h-10 w-full touch-none"
        preserveAspectRatio="none"
        role="img"
        aria-label="Trend over the last minutes"
        onPointerMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const i = Math.round(((e.clientX - r.left) / r.width) * (values.length - 1));
          setHover(Math.min(values.length - 1, Math.max(0, i)));
        }}
        onPointerLeave={() => setHover(null)}
      >
        <path d={d} fill="none" stroke="#94a3b8" strokeWidth={2} vectorEffect="non-scaling-stroke" />
        {Number.isFinite(values[shown]!) && (
          <circle cx={x(shown)} cy={y(values[shown]!)} r={4} fill="#2a78d6" stroke="#fff" strokeWidth={2} vectorEffect="non-scaling-stroke" />
        )}
      </svg>
      {hover != null && Number.isFinite(values[hover]!) && (
        <div className="pointer-events-none absolute -top-7 right-0 rounded-md bg-slate-900 px-2 py-0.5 text-xs text-white">
          at {clock(times[hover] ?? 0)}: {format(values[hover]!)}
        </div>
      )}
    </div>
  );
}

const TILES: { key: string; icon: string; label: string; show: (v: number) => string }[] = [
  { key: "active", icon: "🚗", label: "Cars driving now", show: (v) => fmt(v, 0) },
  { key: "halting", icon: "🛑", label: "Cars standing still", show: (v) => fmt(v, 0) },
  { key: "speed_mean", icon: "💨", label: "Average speed", show: (v) => kmh(v) },
  { key: "arrived", icon: "🏁", label: "Cars that got there", show: (v) => fmt(v, 0) },
];

export function StatTiles({ snap }: { snap: Snapshot }) {
  const times = snap.series.time ?? [];
  return (
    <div className="grid grid-cols-2 gap-3">
      {TILES.map((t) => {
        const v = snap.metrics[t.key];
        return (
          <div key={t.key} className={card}>
            <div className="flex items-center gap-1.5 text-sm text-slate-600">
              <span aria-hidden>{t.icon}</span>
              {t.label}
            </div>
            <div className="mt-1 text-3xl font-semibold text-slate-900">{typeof v === "number" ? t.show(v) : "–"}</div>
            <Sparkline values={snap.series[t.key] ?? []} times={times} format={t.show} />
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------- selection
export function SelectionCard({
  selection,
  geometry,
  controller,
}: {
  selection: Selection | null;
  geometry: Geometry;
  controller: SessionController;
}) {
  if (!selection) {
    return (
      <div className={`${card} text-slate-600`}>
        <div className="mb-1 text-lg font-semibold text-slate-900">👆 Try clicking!</div>
        Click a <b>car</b> on the map to follow it, or a <b>traffic light</b> to see who may go.
      </div>
    );
  }
  if (!selection.available) {
    return (
      <div className={`${card} text-slate-700`}>
        <div className="text-lg font-semibold text-slate-900">🏁 It made it!</div>
        This one has left the town. Click another car to follow.
      </div>
    );
  }
  return selection.kind === "vehicle" ? (
    <CarCard selection={selection} geometry={geometry} />
  ) : (
    <LightCard selection={selection} geometry={geometry} controller={controller} />
  );
}

function CarCard({ selection, geometry }: { selection: Selection; geometry: Geometry }) {
  const d = selection.detail ?? {};
  const kind = VEHICLE[String(d.vclass)] ?? VEHICLE.car!;
  const rows: [string, string, string][] = [
    ["🎯", "Going to", townSide(geometry, d.destination)],
    ["💨", "Speed", kmh(d.speed)],
    ["⏳", "Waited at red lights", seconds(d.waiting_time)],
    ["🛑", "Stopped", `${fmt(d.stops, 0)} time${d.stops === 1 ? "" : "s"}`],
    ["📏", "Has driven", `${fmt(Number(d.distance) / 1000, 2)} km`],
  ];
  const route = remainingRoute(selection);
  return (
    <div className={card}>
      <div className="mb-2 flex items-center gap-2">
        <span className="text-4xl" aria-hidden>{kind.icon}</span>
        <div>
          <div className="text-lg font-semibold text-slate-900">This {kind.name.toLowerCase()}</div>
          <div className="text-xs text-slate-500">{selection.id}</div>
        </div>
      </div>
      {d.vclass === "emergency" && (
        <p className="mb-2 rounded-lg bg-rose-50 p-2 text-sm text-rose-800">
          🚨 Traffic lights turn green for it, and cars move over to let it pass.
        </p>
      )}
      <dl className="space-y-1.5">
        {rows.map(([icon, k, v]) => (
          <div key={k} className="flex items-baseline justify-between gap-2 text-sm">
            <dt className="text-slate-600">
              <span aria-hidden>{icon}</span> {k}
            </dt>
            <dd className="text-right font-semibold text-slate-900">{v}</dd>
          </div>
        ))}
      </dl>
      {route && route.length > 0 && (
        <p className="mt-3 rounded-lg bg-amber-50 p-2 text-sm text-amber-900">
          🟨 The yellow roads on the map show where it will drive next.
        </p>
      )}
    </div>
  );
}

const STATE_WORDS: Record<string, [string, string]> = {
  G: ["🟢", "Go"],
  g: ["🟢", "Go, but give way"],
  y: ["🟡", "Slow down"],
  r: ["🔴", "Stop"],
};

function LightCard({
  selection,
  geometry,
  controller,
}: {
  selection: Selection;
  geometry: Geometry;
  controller: SessionController;
}) {
  const d = selection.detail ?? {};
  const sides = movementSides(geometry, selection.id);
  const states = (d.movement_states as Record<string, string>) ?? {};
  // one row per side: the most permissive state of its movements
  const order = ["G", "g", "y", "r"];
  const bySide = new Map<string, { arrow: string; state: string }>();
  for (const [m, s] of Object.entries(states)) {
    const side = sides.get(m);
    if (!side) continue;
    const cur = bySide.get(side.name);
    if (!cur || order.indexOf(s) < order.indexOf(cur.state)) bySide.set(side.name, { arrow: side.arrow, state: s });
  }
  const phases = (d.phases as { index: number; id: string; green?: string[] }[]) ?? [];
  const phaseSides = (p: { green?: string[] }) => {
    const names = [...new Set((p.green ?? []).map((m) => sides.get(m)?.name).filter(Boolean))] as string[];
    return names.length ? names.join(" & ") : "this phase";
  };
  const b = brain(String(d.controller));
  return (
    <div className={card}>
      <div className="mb-1 text-lg font-semibold text-slate-900">🚦 Traffic light {selection.id}</div>
      <div className="mb-3 text-sm text-slate-600">
        {d.held ? (
          <b className="text-emerald-700">✋ You're in control!</b>
        ) : (
          <>
            Its brain: {b.icon} <b>{b.title}</b>
            {d.remaining != null && <> · changes in {seconds(d.remaining)}</>}
          </>
        )}
      </div>
      {Boolean(d.preempting) && (
        <p className="mb-2 rounded-lg bg-rose-50 p-2 text-sm font-medium text-rose-800">🚑 Keeping it green for an ambulance!</p>
      )}
      <ul className="space-y-1.5">
        {[...bySide.entries()].map(([name, { arrow, state }]) => {
          const [dot, word] = STATE_WORDS[state] ?? ["⚪", state];
          return (
            <li key={name} className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-1.5 text-sm">
              <span className="text-slate-700">
                <span aria-hidden>{arrow}</span> Cars from the {name}
              </span>
              <span className="font-semibold text-slate-900">
                <span aria-hidden>{dot}</span> {word}
              </span>
            </li>
          );
        })}
      </ul>
      {phases.length > 1 && (
        <div className="mt-3">
          <div className="mb-1 text-sm font-medium text-slate-700">✋ You can be the traffic light! Make it green for:</div>
          <div className="flex flex-wrap gap-2">
            {phases.map((p) => (
              <button
                key={p.index}
                onClick={() => controller.send({ type: "hold_phase", intersection: selection.id, phase: p.index })}
                className={`rounded-full px-3 py-1.5 text-sm font-medium ring-1 ${
                  p.index === d.phase_index && d.held
                    ? "bg-emerald-600 text-white ring-emerald-600"
                    : "bg-white text-slate-800 ring-slate-300 hover:bg-emerald-50"
                }`}
              >
                {phaseSides(p)}
              </button>
            ))}
            {Boolean(d.held) && (
              <button
                onClick={() => controller.send({ type: "release_phase", intersection: selection.id })}
                className="rounded-full bg-amber-500 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-600"
              >
                Let it decide by itself again
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------- race tab
const RACE_DURATIONS = [
  { label: "10 minutes", s: 600 },
  { label: "30 minutes", s: 1800 },
];

export function RacePanel({ meta, scenarios }: { meta: Meta; scenarios: ScenarioRef[] }) {
  const rank = (n: string) => (TOWN_ORDER.indexOf(n) + TOWN_ORDER.length + 1) % (TOWN_ORDER.length + 1);
  const bundled = scenarios.filter((s) => s.source === "bundled").map((s) => s.name).sort((a, b) => rank(a) - rank(b));
  const [scenario, setScenario] = useState(bundled.includes("grid_3x3") ? "grid_3x3" : bundled[0] ?? "");
  const choices = brainChoices(meta.controllers);
  const [chosen, setChosen] = useState<string[]>(["fixed_time", "actuated", "max_pressure"].filter((c) => choices.includes(c)));
  const [duration, setDuration] = useState(600);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [science, setScience] = useState(false);
  const seeds = 2;

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.compare({ scenario, controllers: chosen, seeds, duration }));
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className={card}>
        <div className="text-lg font-semibold text-slate-900">🏁 Race the traffic lights</div>
        <p className="mt-1 text-sm text-slate-600">
          The same cars drive through the same town with different traffic-light brains. Which one gets everybody home fastest?
        </p>
      </div>
      <div className={card}>
        <div className="mb-2 text-sm font-medium text-slate-700">1. Pick a town</div>
        <div className="flex flex-wrap gap-2">
          {bundled.map((name) => {
            const t = town(name);
            return (
              <button
                key={name}
                onClick={() => setScenario(name)}
                className={`rounded-full px-3 py-1.5 text-sm ring-1 ${scenario === name ? "bg-sky-600 text-white ring-sky-600" : "bg-white text-slate-800 ring-slate-300 hover:bg-sky-50"}`}
              >
                {t.icon} {t.title}
              </button>
            );
          })}
        </div>
        <div className="mb-2 mt-4 text-sm font-medium text-slate-700">2. Pick the racers (at least two)</div>
        <div className="flex flex-wrap gap-2">
          {choices.map((c) => {
            const b = brain(c);
            const on = chosen.includes(c);
            return (
              <button
                key={c}
                onClick={() => setChosen(on ? chosen.filter((x) => x !== c) : [...chosen, c])}
                aria-pressed={on}
                className={`rounded-full px-3 py-1.5 text-sm ring-1 ${on ? "bg-violet-600 text-white ring-violet-600" : "bg-white text-slate-800 ring-slate-300 hover:bg-violet-50"}`}
              >
                {on ? "✓ " : ""}
                {b.icon} {b.title}
              </button>
            );
          })}
        </div>
        <div className="mb-2 mt-4 text-sm font-medium text-slate-700">3. How long?</div>
        <div className="flex gap-2">
          {RACE_DURATIONS.map((d) => (
            <button
              key={d.s}
              onClick={() => setDuration(d.s)}
              className={`rounded-full px-3 py-1.5 text-sm ring-1 ${duration === d.s ? "bg-slate-800 text-white ring-slate-800" : "bg-white text-slate-800 ring-slate-300"}`}
            >
              {d.label}
            </button>
          ))}
        </div>
        <button
          disabled={busy || chosen.length < 2}
          onClick={run}
          className="mt-4 w-full rounded-full bg-emerald-600 py-3 text-lg font-semibold text-white shadow hover:bg-emerald-500 disabled:opacity-50"
        >
          {busy ? "🏎️ Racing… this takes a little while" : chosen.length < 2 ? "Pick at least two racers" : "🏁 Start the race!"}
        </button>
        {error && <pre className="mt-2 whitespace-pre-wrap rounded-lg bg-red-50 p-2 text-xs text-red-800">{error}</pre>}
      </div>
      {result && <RaceResult result={result} />}
      {result && (
        <button onClick={() => setScience(!science)} className="text-sm text-slate-600 underline">
          {science ? "Hide" : "Show"} the science 🔬
        </button>
      )}
      {result && science && <CompareTable result={result} />}
    </div>
  );
}

const MEDALS = ["🥇", "🥈", "🥉"];

function RaceResult({ result }: { result: CompareResult }) {
  const row = result.metrics.find((m) => m.metric === "travel_time.mean");
  if (!row) return null;
  const ranked = result.controllers
    .map((c) => ({ c, v: row.by_controller[c]?.mean ?? NaN }))
    .filter((x) => Number.isFinite(x.v))
    .sort((a, b) => a.v - b.v);
  const max = Math.max(...ranked.map((x) => x.v), 1);
  const winner = ranked[0];
  const slowest = ranked[ranked.length - 1];
  const faster = winner && slowest ? Math.round((1 - winner.v / slowest.v) * 100) : 0;
  return (
    <div className={card}>
      <div className="text-lg font-semibold text-slate-900">
        {winner ? `${MEDALS[0]} ${brain(winner.c).icon} ${brain(winner.c).title} wins!` : "Results"}
      </div>
      {winner && slowest && winner !== slowest && (
        <p className="mt-1 text-sm text-slate-600">
          Trips were {faster}% shorter than with {brain(slowest.c).icon} {brain(slowest.c).title}.
        </p>
      )}
      <div className="mt-3 text-xs font-medium uppercase tracking-wide text-slate-500">Average time for one trip</div>
      <ul className="mt-2 space-y-2">
        {ranked.map((x, i) => {
          const b = brain(x.c);
          const st = row.by_controller[x.c]!;
          return (
            <li key={x.c} title={`${b.title}: ${fmt(x.v)} s on average (95% range ${fmt(st.ci95[0])}–${fmt(st.ci95[1])} s)`}>
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-800">
                  {MEDALS[i] ?? "  "} {b.icon} {b.title}
                </span>
                <span className="font-semibold text-slate-900">{seconds(x.v)}</span>
              </div>
              <div className="mt-1 h-3 rounded-full bg-slate-100">
                <div className="h-3 rounded-full bg-[#2a78d6]" style={{ width: `${(100 * x.v) / max}%` }} />
              </div>
            </li>
          );
        })}
      </ul>
      <p className="mt-3 text-xs text-slate-500">Shorter bar = cars got home faster.</p>
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

function CompareTable({ result }: { result: CompareResult }) {
  return (
    <div className={`${card} space-y-3`}>
      <p className="text-xs text-slate-500">
        {result.scenario}, {result.seeds.length} seeds × {result.duration ?? "default"} s. Mean; Δ is paired against{" "}
        <b>{result.baseline}</b> (★ = 95% CI excludes 0). Same seeds ⇒ identical demand.
      </p>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-slate-500">
            <th className="py-1 font-medium">Metric</th>
            {result.controllers.map((c) => (
              <th key={c} className="py-1 text-right font-medium">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {result.metrics.map((row) => (
            <tr key={row.metric} className="border-t border-slate-100">
              <td className="py-1 text-slate-600">{LABELS[row.metric] ?? row.metric}</td>
              {result.controllers.map((c) => {
                const delta = row.vs_baseline[c];
                return (
                  <td key={c} className="py-1 text-right text-slate-900">
                    {fmt(row.by_controller[c]?.mean)}
                    {delta?.delta_pct != null && (
                      <span className="block text-[10px] text-slate-500">
                        {delta.delta_pct > 0 ? "+" : ""}
                        {fmt(delta.delta_pct)}%{delta.significant ? "★" : ""}
                      </span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------- grown-ups
export function TechnicalDetails({ snap, selection }: { snap: Snapshot; selection: Selection | null }) {
  const m = snap.metrics;
  const rows: [string, string][] = [
    ["Vehicles on network", fmt(m.active, 0)],
    ["Mean speed", `${fmt(m.speed_mean)} m/s`],
    ["Network queue (time-mean)", `${fmt(m.queue_mean)} veh`],
    ["Throughput", `${fmt(m.throughput_vph, 0)} veh/h`],
    ["Generated / arrived", `${fmt(m.generated, 0)} / ${fmt(m.arrived, 0)}`],
    ["Waiting to enter", fmt(m.backlog, 0)],
  ];
  return (
    <div className={`${card} space-y-3 text-xs`}>
      <div className="text-sm font-semibold text-slate-900">Metrics (10-second samples)</div>
      <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-slate-500">{k}</dt>
            <dd className="text-right font-mono text-slate-900">{v}</dd>
          </div>
        ))}
      </dl>
      {selection?.available && (
        <>
          <div className="text-sm font-semibold text-slate-900">Selected {selection.kind}: {selection.id}</div>
          <pre className="max-h-64 overflow-auto rounded-lg bg-slate-50 p-2 text-[11px] text-slate-700">
            {JSON.stringify(selection.detail, null, 1)}
          </pre>
        </>
      )}
    </div>
  );
}
