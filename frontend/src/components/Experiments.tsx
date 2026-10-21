// Experiments: many simulations at once (traffic-light strategies x repeats with different
// random traffic), kept in the workspace. Follow their progress, compare the results with
// 95 % ranges, see how traffic developed over time, watch recorded runs, download it all.

import { useEffect, useState } from "react";
import {
  api,
  downloads,
  type CompareRow,
  type ExperimentComparison,
  type ExperimentDoc,
  type ExperimentSeries,
  type ExperimentSummary,
  type Meta,
  type ScenarioRef,
} from "../lib/api";
import { SERIES_COLOR } from "../lib/colors";
import { brain, brainChoices, clock, isBrain, kmh, seconds, town, TOWN_ORDER } from "../lib/friendly";
import { CompareTable, fmt } from "./Panels";
import { Logo, pill } from "./StartScreen";

const card = "rounded-2xl bg-white p-4 shadow-sm ring-1 ring-slate-200";
const ACTIVE = new Set(["queued", "running"]);
const MAP_PLAN = "map-plan";
const MAX_SERIES = SERIES_COLOR.length;
const INK = "#0f172a";
const INK_2 = "#475569";
const GRID = "#e2e8f0";
const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));
const num = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);

const STATUS: Record<string, [string, string]> = {
  draft: ["📝 Not started", "bg-slate-100 text-slate-700"],
  queued: ["⏳ Waiting its turn", "bg-slate-100 text-slate-700"],
  running: ["🏃 Running", "bg-sky-100 text-sky-800"],
  completed: ["✅ Finished", "bg-emerald-100 text-emerald-800"],
  partial: ["⚠️ Partly finished", "bg-amber-100 text-amber-800"],
  failed: ["❌ Failed", "bg-red-100 text-red-800"],
  cancelled: ["⏹ Stopped", "bg-slate-100 text-slate-700"],
  interrupted: ["⏸ Interrupted (the server restarted)", "bg-amber-100 text-amber-800"],
};
const RUN_STATUS: Record<string, string> = {
  pending: "⏳ waiting",
  running: "🏃 running",
  completed: "✅ done",
  failed: "❌ failed",
  cancelled: "⏹ stopped",
  interrupted: "⏸ interrupted",
};

function Status({ status }: { status: string }) {
  const [text, cls] = STATUS[status] ?? [status, "bg-slate-100 text-slate-700"];
  return <span className={`whitespace-nowrap rounded-full px-3 py-1 text-xs font-semibold ${cls}`}>{text}</span>;
}

/** A variant's name in plain words (UI-made variants are named after their strategy). */
function variantLabel(name: string): string {
  if (name === MAP_PLAN) return "✨ As the map is set up";
  if (isBrain(name)) return `${brain(name).icon} ${brain(name).title}`;
  return name;
}

function mapLabel(scenario: string | null): string {
  if (!scenario) return "a generated map";
  if (scenario.includes("/")) return `🗺️ ${scenario.split("/").pop()!.replace(/\.json$/, "")}`;
  const t = town(scenario);
  return `${t.icon} ${t.title}`;
}

/** ``work``: the share of all the simulating done (finished runs count 1, running ones their
 * own progress); shown while the experiment runs. */
function Progress({ p, work }: { p: ExperimentSummary["progress"]; work?: number }) {
  const total = p.total || 1;
  const seg = (n: number | undefined, cls: string) =>
    n ? <div className={cls} style={{ width: `${(100 * n) / total}%` }} /> : null;
  return (
    <div>
      <div className="flex h-2.5 gap-0.5 overflow-hidden rounded-full bg-slate-100" aria-hidden>
        {seg(p.completed, "bg-emerald-500")}
        {seg(p.failed, "bg-red-500")}
        {seg(p.running, "animate-pulse bg-sky-500")}
      </div>
      <div className="mt-1 text-xs text-slate-600">
        {p.completed ?? 0} of {p.total} simulations finished
        {p.failed ? ` · ${p.failed} failed` : ""}
        {p.running ? ` · ${p.running} running` : ""}
        {p.running && work != null ? ` · about ${Math.floor(100 * work)} % of the work done` : ""}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------- page
export function ExperimentsPage({
  meta,
  scenarios,
  open,
  onOpen,
  onBack,
  onWatch,
}: {
  meta: Meta;
  scenarios: ScenarioRef[];
  /** The experiment shown (``null``: the list). */
  open: string | null;
  onOpen: (id: string | null) => void;
  onBack: () => void;
  onWatch: (runId: string, label: string) => Promise<void>;
}) {
  return (
    <div className="mx-auto max-w-5xl space-y-6 p-4 md:p-8">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <Logo />
        <button onClick={onBack} className={pill(false)}>← Back to the start</button>
      </header>
      {open ? (
        <ExperimentDetail key={open} id={open} onClose={() => onOpen(null)} onWatch={onWatch} />
      ) : (
        <ExperimentList meta={meta} scenarios={scenarios} onOpen={onOpen} />
      )}
    </div>
  );
}

function ExperimentList({ meta, scenarios, onOpen }: { meta: Meta; scenarios: ScenarioRef[]; onOpen: (id: string) => void }) {
  const [items, setItems] = useState<ExperimentSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    let alive = true;
    let timer = 0;
    const load = () =>
      api.experiments()
        .then((r) => {
          if (!alive) return;
          setItems(r.items);
          if (r.items.some((e) => ACTIVE.has(e.status))) timer = window.setTimeout(load, 1500);
        })
        .catch((e) => alive && setError(msg(e)));
    load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);

  return (
    <>
      <section className="space-y-2">
        <h1 className="text-2xl font-bold text-slate-900">🧪 Experiments</h1>
        <p className="max-w-3xl text-sm text-slate-600">
          Run each traffic-light strategy several times on the same map, with different random traffic each time, and find
          out which one is really better rather than just lucky. Experiments keep running while you do other things, and
          their results stay here.
        </p>
      </section>
      {creating ? (
        <NewExperiment meta={meta} scenarios={scenarios} onCreated={onOpen} onCancel={() => setCreating(false)} />
      ) : (
        <button onClick={() => setCreating(true)} className="rounded-full bg-emerald-600 px-6 py-3 text-lg font-semibold text-white shadow hover:bg-emerald-500">
          ➕ New experiment
        </button>
      )}
      {error && <p className="rounded-lg bg-red-50 p-3 text-sm text-red-800">{error}</p>}
      <section className="space-y-3">
        {items?.length === 0 && <p className="text-sm text-slate-500">No experiments yet. Make one with the button above.</p>}
        {items?.map((e) => (
          <button key={e.id} onClick={() => onOpen(e.id)} className={`${card} block w-full text-left transition hover:ring-sky-300`}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="font-semibold text-slate-900">{e.name}</span>
              <Status status={e.status} />
            </div>
            <div className="mt-1 text-sm text-slate-600">
              {mapLabel(e.scenario)} · {e.variants.length} strateg{e.variants.length === 1 ? "y" : "ies"} × {e.seeds.length} repeat
              {e.seeds.length === 1 ? "" : "s"} · {new Date(e.created_at).toLocaleString()}
            </div>
            <div className="mt-2">
              <Progress p={e.progress} />
            </div>
          </button>
        ))}
      </section>
    </>
  );
}

// ---------------------------------------------------------------------------- new
const DURATIONS = [
  { s: 600, label: "10 minutes" },
  { s: 1800, label: "30 minutes" },
  { s: 3600, label: "1 hour" },
];
const TRAFFIC = [
  { f: 0.5, label: "Light" },
  { f: 1, label: "Normal" },
  { f: 1.5, label: "Heavy" },
  { f: 2, label: "Very heavy" },
];
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^[^a-z0-9]+/, "").slice(0, 60) || "experiment";

function NewExperiment({
  meta,
  scenarios,
  onCreated,
  onCancel,
}: {
  meta: Meta;
  scenarios: ScenarioRef[];
  onCreated: (id: string) => void;
  onCancel: () => void;
}) {
  const maps = [
    ...TOWN_ORDER.flatMap((n) => scenarios.filter((s) => s.source === "bundled" && s.name === n)),
    ...scenarios.filter((s) => s.source === "bundled" && !TOWN_ORDER.includes(s.name)),
    ...scenarios.filter((s) => s.source === "workspace"),
  ];
  const [map, setMap] = useState<ScenarioRef | undefined>(maps.find((s) => s.name === "grid_3x3") ?? maps[0]);
  const choices = ["", ...brainChoices(meta.controllers)];
  const [chosen, setChosen] = useState<string[]>(["", "actuated", "max_pressure"].filter((c) => choices.includes(c)));
  const [repeats, setRepeats] = useState(3);
  const [duration, setDuration] = useState(600);
  const [traffic, setTraffic] = useState(1);
  const [record, setRecord] = useState(true);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const strategies = choices.filter((c) => chosen.includes(c));
  const auto = slug(`${map?.name ?? "map"}-${strategies.length}-strategies`);

  const create = async () => {
    if (!map) return;
    setBusy(true);
    setError(null);
    const where = map.source === "bundled" ? { bundled: map.name } : { path: `scenarios/${map.name}.json` };
    const spec = {
      format: "urbanflow.experiment",
      version: "1.0",
      name: name || auto,
      description: `${TRAFFIC.find((t) => t.f === traffic)?.label} traffic, ${DURATIONS.find((d) => d.s === duration)?.label} per simulation`,
      scenario: { ...where, demand_scale: traffic },
      base_config: { duration },
      variants: strategies.map((c) => (c ? { name: c, policy: { kind: "controller", controller: c } } : { name: MAP_PLAN })),
      seeds: { start: 0, count: repeats },
      record_replay: record ? "first_seed" : false,
    };
    try {
      onCreated((await api.createExperiment(spec, true)).id);
    } catch (e) {
      setError(msg(e));
      setBusy(false);
    }
  };

  const label = "mb-2 mt-4 block text-sm font-medium text-slate-700";
  return (
    <section className={`${card} ring-2 ring-emerald-300`}>
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900">➕ New experiment</h2>
        <button onClick={onCancel} className="text-slate-500 hover:text-slate-800" aria-label="Close">✕</button>
      </div>
      <span className={label}>1. Map</span>
      <div className="flex flex-wrap gap-2">
        {maps.map((s) => {
          const t = s.source === "workspace" ? { icon: "🗺️", title: s.name } : town(s.name);
          const on = map === s;
          return (
            <button key={`${s.source}:${s.name}`} onClick={() => setMap(s)} aria-pressed={on} className={pill(on)}>
              {t.icon} {t.title}
            </button>
          );
        })}
      </div>
      <span className={label}>2. Traffic-light strategies to test (the first one is what the others are compared with)</span>
      <div className="flex flex-wrap gap-2">
        {choices.map((c) => {
          const on = chosen.includes(c);
          return (
            <button key={c || "map"} onClick={() => setChosen(on ? chosen.filter((x) => x !== c) : [...chosen, c])} aria-pressed={on}
              className={pill(on, "bg-violet-600 text-white ring-violet-600")}>
              {on ? "✓ " : ""}
              {variantLabel(c || MAP_PLAN)}
            </button>
          );
        })}
      </div>
      <span className={label}>3. How much traffic?</span>
      <div className="flex flex-wrap gap-2">
        {TRAFFIC.map((t) => (
          <button key={t.f} onClick={() => setTraffic(t.f)} aria-pressed={traffic === t.f} className={pill(traffic === t.f)}>{t.label}</button>
        ))}
      </div>
      <span className={label}>4. Simulated time for each simulation</span>
      <div className="flex flex-wrap gap-2">
        {DURATIONS.map((d) => (
          <button key={d.s} onClick={() => setDuration(d.s)} aria-pressed={duration === d.s} className={pill(duration === d.s)}>{d.label}</button>
        ))}
      </div>
      <label className={label}>
        5. Repeats: each strategy runs{" "}
        <input type="number" min={1} max={20} value={repeats} onChange={(e) => setRepeats(Math.min(20, Math.max(1, Math.round(+e.target.value || 1))))}
          className="mx-1 w-16 rounded-lg border border-slate-300 p-1 text-center" />{" "}
        times, with different random traffic (the same traffic for every strategy). More repeats give a surer answer.
      </label>
      <label className="mt-3 flex items-center gap-2 text-sm text-slate-700">
        <input type="checkbox" checked={record} onChange={(e) => setRecord(e.target.checked)} className="h-4 w-4" />
        ⏺ Record the first repeat of each strategy so I can watch it
      </label>
      <label className={label}>
        Name
        <input value={name} onChange={(e) => setName(slug(e.target.value).slice(0, 64))} placeholder={auto}
          className="mt-1 block w-full max-w-md rounded-lg border border-slate-300 p-1.5 font-normal" />
      </label>
      <button disabled={busy || !map || strategies.length === 0} onClick={create}
        className="mt-4 rounded-full bg-emerald-600 px-6 py-3 font-semibold text-white shadow hover:bg-emerald-500 disabled:opacity-50">
        {busy ? "Starting…" : strategies.length === 0 ? "Choose at least one strategy" : `▶ Run ${strategies.length * repeats} simulations`}
      </button>
      {strategies.length === 1 && <p className="mt-2 text-xs text-slate-500">Tip: choose two or more strategies to compare them.</p>}
      {error && <pre className="mt-3 whitespace-pre-wrap rounded-lg bg-red-50 p-3 font-sans text-sm text-red-800">{error}</pre>}
    </section>
  );
}

// ---------------------------------------------------------------------------- detail
function ExperimentDetail({ id, onClose, onWatch }: { id: string; onClose: () => void; onWatch: (runId: string, label: string) => Promise<void> }) {
  const [doc, setDoc] = useState<ExperimentDoc | null>(null);
  const [cmp, setCmp] = useState<ExperimentComparison | null>(null);
  const [baseline, setBaseline] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let alive = true;
    let timer = 0;
    const load = () =>
      api.experiment(id)
        .then((d) => {
          if (!alive) return;
          setDoc(d);
          if (ACTIVE.has(d.status)) timer = window.setTimeout(load, 1000);
        })
        .catch((e) => alive && setError(msg(e)));
    load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [id, reload]);

  const done = doc?.progress.completed ?? 0;
  const base = baseline ?? doc?.baseline ?? "";
  useEffect(() => {
    if (!done || !base) return;
    api.experimentCompare(id, base).then(setCmp).catch((e) => setError(msg(e)));
  }, [id, done, base]);

  if (!doc) return <p className="text-slate-500">{error ?? "Loading…"}</p>;
  const active = ACTIVE.has(doc.status);
  const unfinished = doc.progress.total - done;
  const act = async (fn: () => Promise<unknown>) => {
    setError(null);
    try {
      await fn();
      setReload((n) => n + 1);
    } catch (e) {
      setError(msg(e));
    }
  };
  const remove = async () => {
    if (!confirm(`Delete the experiment "${doc.name}" and all its results?`)) return;
    try {
      await api.deleteExperiment(id);
      onClose();
    } catch (e) {
      setError(msg(e));
    }
  };

  return (
    <>
      <button onClick={onClose} className="text-sm text-sky-700 underline">← All experiments</button>
      <section className={`${card} space-y-3`}>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h1 className="text-2xl font-bold text-slate-900">🧪 {doc.name}</h1>
          <Status status={doc.status} />
        </div>
        <p className="text-sm text-slate-600">
          {mapLabel(doc.scenario)} · {doc.variants.length} strateg{doc.variants.length === 1 ? "y" : "ies"} × {doc.seeds.length} repeat
          {doc.seeds.length === 1 ? "" : "s"}
          {doc.description ? ` · ${doc.description}` : ""}
        </p>
        <Progress p={doc.progress} work={doc.runs.reduce((a, r) => a + (r.status === "completed" ? 1 : r.status === "running" ? r.progress : 0), 0) / (doc.progress.total || 1)} />
        <div className="flex flex-wrap gap-2">
          {active && <button onClick={() => act(() => api.cancelExperiment(id))} className={pill(false)}>⏹ Stop</button>}
          {!active && unfinished > 0 && (
            <button onClick={() => act(() => api.runExperiment(id))} className={pill(true, "bg-emerald-600 text-white ring-emerald-600")}>
              {doc.status === "draft" ? "▶ Run" : `▶ Run the ${unfinished} unfinished`}
            </button>
          )}
          {!active && <button onClick={remove} className={pill(false)}>🗑 Delete</button>}
        </div>
        {error && <pre className="whitespace-pre-wrap rounded-lg bg-red-50 p-3 font-sans text-sm text-red-800">{error}</pre>}
      </section>

      {done > 0 && cmp && (
        <Results doc={doc} cmp={cmp} baseline={base} onBaseline={setBaseline} />
      )}
      {done > 0 && <OverTime id={id} variants={doc.variants} done={done} />}
      {done === 0 && !active && doc.status !== "draft" && <p className="text-sm text-slate-600">No simulation finished, so there are no results.</p>}
      {done === 0 && active && <p className="text-sm text-slate-600">Results appear here as soon as the first simulations finish.</p>}
      <Runs doc={doc} onWatch={onWatch} />
      <section className="flex flex-wrap gap-x-5 gap-y-2 text-sm">
        <span className="text-slate-600">⬇ Download:</span>
        <a className="text-sky-700 underline" href={downloads.experiment(id, "comparison", "csv")}>comparison (CSV)</a>
        <a className="text-sky-700 underline" href={downloads.experiment(id, "comparison", "json")}>comparison (JSON)</a>
        <a className="text-sky-700 underline" href={downloads.experiment(id, "runs", "csv")}>every simulation's results (CSV)</a>
        <a className="text-sky-700 underline" href={downloads.experiment(id, "runs", "json")}>(JSON)</a>
      </section>
    </>
  );
}

// ---------------------------------------------------------------------------- results
const MEASURES: { key: string; label: string; show: (v: number) => string; less: string; more: string }[] = [
  { key: "travel_time.mean", label: "Trip time", show: seconds, less: "shorter trips", more: "longer trips" },
  { key: "delay.mean", label: "Time lost in traffic", show: seconds, less: "less time lost", more: "more time lost" },
  { key: "waiting_time_mean", label: "Time standing still", show: seconds, less: "less waiting", more: "more waiting" },
  { key: "stops_mean", label: "Stops per car", show: (v) => fmt(v, 1), less: "fewer stops", more: "more stops" },
  { key: "throughput_vph", label: "Cars arriving per hour", show: (v) => fmt(v, 0), less: "fewer cars arriving", more: "more cars arriving" },
  { key: "queue.mean_total_veh", label: "Cars in queues", show: (v) => fmt(v, 0), less: "shorter queues", more: "longer queues" },
];

function Results({ doc, cmp, baseline, onBaseline }: { doc: ExperimentDoc; cmp: ExperimentComparison; baseline: string; onBaseline: (b: string) => void }) {
  const [key, setKey] = useState(MEASURES[0]!.key);
  const [numbers, setNumbers] = useState(false);
  const measure = MEASURES.find((m) => m.key === key)!;
  const row = cmp.metrics.find((m) => m.metric === key);
  const asTable = {
    scenario: doc.scenario ?? doc.name,
    controllers: cmp.variants,
    baseline: cmp.baseline,
    seeds: cmp.seeds,
    duration: typeof doc.spec.base_config.duration === "number" ? doc.spec.base_config.duration : null,
    metrics: cmp.metrics,
  };
  return (
    <section className={`${card} space-y-3`}>
      <h2 className="text-lg font-semibold text-slate-900">📊 Results</h2>
      <div className="flex flex-wrap gap-2">
        {MEASURES.filter((m) => cmp.metrics.some((r) => r.metric === m.key)).map((m) => (
          <button key={m.key} onClick={() => setKey(m.key)} aria-pressed={key === m.key} className={pill(key === m.key)}>{m.label}</button>
        ))}
      </div>
      {row && <Verdict row={row} variants={cmp.variants} baseline={cmp.baseline} measure={measure} />}
      {row && <BarChart row={row} variants={cmp.variants} baseline={cmp.baseline} show={measure.show} label={measure.label} />}
      <p className="text-xs text-slate-500">
        Bars are averages over the repeats; the thin line across each bar is its 95 % range (where the true average most
        likely is).
        {cmp.variants.length > MAX_SERIES ? ` The chart shows the first ${MAX_SERIES} strategies; the numbers below list all of them.` : ""}
        {cmp.missing.length ? ` Not included yet: ${cmp.missing.length} unfinished simulation${cmp.missing.length > 1 ? "s" : ""}.` : ""}
      </p>
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <label className="text-slate-600">
          Compare against{" "}
          <select value={baseline} onChange={(e) => onBaseline(e.target.value)} className="rounded-lg border border-slate-300 bg-white p-1">
            {cmp.variants.map((v) => <option key={v} value={v}>{variantLabel(v)}</option>)}
          </select>
        </label>
        <button onClick={() => setNumbers(!numbers)} className="text-slate-600 underline">{numbers ? "Hide" : "Show"} all the numbers</button>
      </div>
      {numbers && <CompareTable result={asTable} />}
    </section>
  );
}

function Verdict({ row, variants, baseline, measure }: { row: CompareRow; variants: string[]; baseline: string; measure: (typeof MEASURES)[number] }) {
  const sign = row.direction === "max" ? -1 : 1;
  const ranked = variants
    .filter((v) => num(row.by_controller[v]?.mean))
    .sort((a, b) => sign * (row.by_controller[a]!.mean! - row.by_controller[b]!.mean!));
  const best = ranked[0];
  if (!best || ranked.length < 2) return null;
  if (best === baseline) {
    const close = variants.filter((v) => v !== baseline && row.vs_baseline[v] && !row.vs_baseline[v]!.significant);
    return (
      <p className="text-base text-slate-800">
        🥇 <b>{variantLabel(best)}</b> did best on {measure.label.toLowerCase()}
        {close.length ? `, though ${close.map(variantLabel).join(", ")} ${close.length > 1 ? "were" : "was"} too close to call.` : ", clearly ahead of the others."}
      </p>
    );
  }
  const d = row.vs_baseline[best];
  const pct = d?.delta_pct;
  return (
    <p className="text-base text-slate-800">
      🥇 <b>{variantLabel(best)}</b> did best on {measure.label.toLowerCase()}
      {num(pct)
        ? Math.abs(pct) < 0.5
          ? `, about the same as ${variantLabel(baseline)}`
          : `: ${fmt(Math.abs(pct), 0)} % ${pct < 0 ? measure.less : measure.more} than ${variantLabel(baseline)}`
        : ""}
      .{" "}
      {d?.significant
        ? "That held across the repeats, so it is a real difference."
        : `The repeats disagree, so it may be luck: try more repeats (${d?.n_pairs ?? 0} so far).`}
    </p>
  );
}

function BarChart({ row, variants, baseline, show, label }: { row: CompareRow; variants: string[]; baseline: string; show: (v: number) => string; label: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const shown = variants.slice(0, MAX_SERIES);
  const W = 640, labelW = 260, valueW = 96, rowH = 34, barH = 18;
  const H = shown.length * rowH + 4;
  const top = (i: number) => i * rowH + (rowH - barH) / 2;
  const max = Math.max(1e-9, ...shown.flatMap((v) => [row.by_controller[v]?.mean, row.by_controller[v]?.ci95[1]].filter(num)));
  const x = (v: number) => labelW + (Math.max(0, v) / max) * (W - labelW - valueW);
  const bar = (y: number, w: number) => {
    const r = Math.min(4, w / 2, barH / 2);
    return `M${labelW},${y}H${labelW + w - r}Q${labelW + w},${y} ${labelW + w},${y + r}V${y + barH - r}Q${labelW + w},${y + barH} ${labelW + w - r},${y + barH}H${labelW}Z`;
  };
  const h = hover == null ? null : shown[hover]!;
  const hs = h ? row.by_controller[h] : undefined;
  const hd = h ? row.vs_baseline[h] : undefined;
  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${label} by strategy`}>
        {shown.map((v, i) => {
          const st = row.by_controller[v];
          const y = top(i);
          const [lo, hi] = st?.ci95 ?? [null, null];
          const end = Math.max(num(st?.mean) ? st!.mean! : 0, num(hi) ? hi : 0);
          return (
            <g key={v} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
              <rect x={0} y={i * rowH} width={W} height={rowH} fill={hover === i ? "#f1f5f9" : "transparent"} rx={6} />
              <text x={labelW - 10} y={y + barH / 2} textAnchor="end" dominantBaseline="middle" fontSize={13} fill={INK}>
                {variantLabel(v)}{v === baseline ? " (reference)" : ""}
              </text>
              {num(st?.mean) && <path d={bar(y, x(st!.mean!) - labelW)} fill={SERIES_COLOR[i]} />}
              {num(lo) && num(hi) && (
                <g stroke={INK} strokeWidth={1.5} strokeLinecap="round">
                  <line x1={x(lo)} x2={x(hi)} y1={y + barH / 2} y2={y + barH / 2} />
                  <line x1={x(lo)} x2={x(lo)} y1={y + 4} y2={y + barH - 4} />
                  <line x1={x(hi)} x2={x(hi)} y1={y + 4} y2={y + barH - 4} />
                </g>
              )}
              <text x={x(end) + 8} y={y + barH / 2} dominantBaseline="middle" fontSize={13} fontWeight={600} fill={INK}>
                {num(st?.mean) ? show(st!.mean!) : "no result yet"}
              </text>
            </g>
          );
        })}
        <line x1={labelW} x2={labelW} y1={0} y2={H} stroke="#94a3b8" />
      </svg>
      {h && hs && (
        <div className="pointer-events-none absolute right-2 z-10 rounded-lg bg-slate-900 px-3 py-2 text-xs text-white shadow-lg" style={{ top: `${(100 * top(hover!)) / H}%`, transform: "translateY(-110%)" }}>
          <div className="font-semibold">{variantLabel(h)}</div>
          <div>Average {num(hs.mean) ? show(hs.mean) : "–"} over {hs.values.filter(num).length} repeats</div>
          {num(hs.ci95[0]) && num(hs.ci95[1]) && <div>95 % range {show(hs.ci95[0])} – {show(hs.ci95[1])}</div>}
          {hd && num(hd.delta_pct) && (
            <div>
              {hd.delta_pct > 0 ? "+" : ""}
              {fmt(hd.delta_pct, 1)} % vs {variantLabel(baseline)} ({hd.significant ? "a real difference" : "could be luck"})
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------- over time
const COLUMNS: { key: string; label: string; show: (v: number) => string }[] = [
  { key: "active", label: "Cars driving", show: (v) => fmt(v, 0) },
  { key: "halting", label: "Cars standing still", show: (v) => fmt(v, 0) },
  { key: "speed_mean", label: "Average speed", show: kmh },
  { key: "queue_mean", label: "Cars in queues", show: (v) => fmt(v, 0) },
  { key: "backlog", label: "Cars waiting to enter the map", show: (v) => fmt(v, 0) },
];

function OverTime({ id, variants, done }: { id: string; variants: string[]; done: number }) {
  const [col, setCol] = useState("active");
  const [series, setSeries] = useState<ExperimentSeries[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.experimentTimeseries(id, col)
      .then((r) => {
        setSeries(r.series);
        setError(null);
      })
      .catch((e) => setError(msg(e)));
  }, [id, col, done]);
  const column = COLUMNS.find((c) => c.key === col)!;
  return (
    <section className={`${card} space-y-3`}>
      <h2 className="text-lg font-semibold text-slate-900">📈 Over time</h2>
      <div className="flex flex-wrap gap-2">
        {COLUMNS.map((c) => (
          <button key={c.key} onClick={() => setCol(c.key)} aria-pressed={col === c.key} className={pill(col === c.key)}>{c.label}</button>
        ))}
      </div>
      {error && <p className="text-sm text-red-700">{error}</p>}
      {series && series.length > 0 && <LineChart series={series} variants={variants} show={column.show} label={column.label} />}
      {series?.length === 0 && <p className="text-sm text-slate-500">No finished simulation has this table yet.</p>}
      <p className="text-xs text-slate-500">Lines are averages over the repeats; the shaded band around each is its 95 % range.</p>
    </section>
  );
}

/** Round axis ticks from 0 to just above ``v`` (about four steps of 1, 2, 2.5 or 5 x 10^k). */
function ticks(v: number): number[] {
  const raw = v / 4;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 2.5, 5, 10].find((m) => m * p >= raw) ?? 10) * p;
  return Array.from({ length: Math.ceil(v / step - 1e-9) + 1 }, (_, i) => i * step);
}

/** Index of the sample nearest to time ``t`` (times ascending). */
function nearest(times: number[], t: number): number {
  let lo = 0, hi = times.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid]! < t) lo = mid;
    else hi = mid;
  }
  return Math.abs(times[lo]! - t) <= Math.abs(times[hi]! - t) ? lo : hi;
}

function LineChart({ series, variants, show, label }: { series: ExperimentSeries[]; variants: string[]; show: (v: number) => string; label: string }) {
  const [hoverT, setHoverT] = useState<number | null>(null);
  const shown = series.filter((s) => variants.indexOf(s.variant) < MAX_SERIES);
  const direct = shown.length <= 4;
  const W = 640, H = 250, L = 58, R = direct ? 150 : 16, T = 12, B = 26;
  const tMax = Math.max(1, ...shown.flatMap((s) => s.time));
  const ticksY = ticks(Math.max(1e-9, ...shown.flatMap((s) => [...s.mean, ...s.hi].filter(num))));
  const yMax = ticksY[ticksY.length - 1]!;
  const x = (t: number) => L + (t / tMax) * (W - L - R);
  const y = (v: number) => T + (1 - v / yMax) * (H - T - B);
  const color = (v: string) => SERIES_COLOR[variants.indexOf(v)]!;
  const line = (s: ExperimentSeries) => {
    let d = "", pen = false;
    s.mean.forEach((v, i) => {
      if (num(v)) {
        d += `${pen ? "L" : "M"}${x(s.time[i]!)},${y(v)}`;
        pen = true;
      } else pen = false;
    });
    return d;
  };
  const bands = (s: ExperimentSeries) => {
    const out: string[] = [];
    let run: number[] = [];
    const flush = () => {
      if (run.length > 1) {
        const upper = run.map((i) => `${x(s.time[i]!)},${y(s.hi[i]!)}`);
        const lower = run.slice().reverse().map((i) => `${x(s.time[i]!)},${y(s.lo[i]!)}`);
        out.push(`M${upper.join("L")}L${lower.join("L")}Z`);
      }
      run = [];
    };
    s.time.forEach((_, i) => (num(s.lo[i]) && num(s.hi[i]) ? run.push(i) : flush()));
    flush();
    return out;
  };
  // End labels, pushed apart so they never overlap.
  const ends = shown
    .map((s) => {
      const i = s.mean.map(num).lastIndexOf(true);
      return { s, i, y: i < 0 ? 0 : y(s.mean[i]!) };
    })
    .filter((e) => e.i >= 0)
    .sort((a, b) => a.y - b.y);
  for (let k = 1; k < ends.length; k++) ends[k]!.y = Math.max(ends[k]!.y, ends[k - 1]!.y + 14);
  const ticksT = [0, 0.25, 0.5, 0.75, 1].map((f) => f * tMax);
  const ref = shown[0]!;
  const hi = hoverT == null ? null : nearest(ref.time, hoverT);
  const ht = hi == null ? null : ref.time[hi]!;

  return (
    <div className="space-y-2">
      {shown.length > 1 && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-700">
          {shown.map((s) => (
            <span key={s.variant} className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 rounded" style={{ background: color(s.variant) }} />
              {variantLabel(s.variant)}
            </span>
          ))}
        </div>
      )}
      <div className="relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full touch-none"
          role="img"
          aria-label={`${label} over time`}
          onPointerMove={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            const px = ((e.clientX - r.left) / r.width) * W;
            setHoverT(Math.min(1, Math.max(0, (px - L) / (W - L - R))) * tMax);
          }}
          onPointerLeave={() => setHoverT(null)}
        >
          {ticksY.map((v) => (
            <g key={v}>
              <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke={GRID} />
              <text x={L - 6} y={y(v)} textAnchor="end" dominantBaseline="middle" fontSize={11} fill={INK_2}>{show(v)}</text>
            </g>
          ))}
          {ticksT.map((t) => (
            <text key={t} x={x(t)} y={H - 8} textAnchor="middle" fontSize={11} fill={INK_2}>{clock(t)}</text>
          ))}
          {shown.map((s) => bands(s).map((d, k) => <path key={`${s.variant}${k}`} d={d} fill={color(s.variant)} opacity={0.14} />))}
          {shown.map((s) => (
            <path key={s.variant} d={line(s)} fill="none" stroke={color(s.variant)} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {direct &&
            ends.map((e) => (
              <text key={e.s.variant} x={x(e.s.time[e.i]!) + 8} y={e.y} dominantBaseline="middle" fontSize={12} fill={INK}>
                {variantLabel(e.s.variant)}
              </text>
            ))}
          {ht != null && (
            <g>
              <line x1={x(ht)} x2={x(ht)} y1={T} y2={H - B} stroke="#94a3b8" />
              {shown.map((s) => {
                const v = s.mean[nearest(s.time, ht)];
                return num(v) ? <circle key={s.variant} cx={x(ht)} cy={y(v)} r={4} fill={color(s.variant)} stroke="#fff" strokeWidth={2} /> : null;
              })}
            </g>
          )}
        </svg>
        {ht != null && (
          <div
            className="pointer-events-none absolute top-0 z-10 rounded-lg bg-slate-900 px-3 py-2 text-xs text-white shadow-lg"
            style={x(ht) > W / 2 ? { right: `${100 - (100 * x(ht)) / W + 1}%` } : { left: `${(100 * x(ht)) / W + 1}%` }}
          >
            <div className="font-semibold">after {clock(ht)}</div>
            {shown.map((s) => {
              const i = nearest(s.time, ht);
              const v = s.mean[i];
              return (
                <div key={s.variant} className="flex items-center gap-1.5 whitespace-nowrap">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: color(s.variant) }} />
                  {variantLabel(s.variant)}: {num(v) ? show(v) : "–"}
                  {num(s.lo[i]) && num(s.hi[i]) ? <span className="opacity-70"> ({show(s.lo[i]!)}–{show(s.hi[i]!)})</span> : null}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------- runs
function Runs({ doc, onWatch }: { doc: ExperimentDoc; onWatch: (runId: string, label: string) => Promise<void> }) {
  const [error, setError] = useState<string | null>(null);
  const failed = doc.runs.some((r) => r.status === "failed");
  const watch = (runId: string, label: string) => onWatch(runId, label).catch((e) => setError(msg(e)));
  return (
    <details className={card} open={failed || doc.runs.some((r) => r.replay)}>
      <summary className="cursor-pointer text-lg font-semibold text-slate-900">🧾 Every simulation ({doc.runs.length})</summary>
      {error && <p className="mt-2 text-sm text-red-700">{error}</p>}
      <div className="mt-3 max-h-96 overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-white">
            <tr className="text-left text-xs text-slate-500">
              <th className="py-1 font-medium">Strategy</th>
              <th className="py-1 font-medium">Repeat</th>
              <th className="py-1 font-medium">Status</th>
              <th className="py-1 text-right font-medium">Took</th>
              <th className="py-1" />
            </tr>
          </thead>
          <tbody>
            {doc.runs.map((r) => {
              const label = `${doc.name}: ${variantLabel(r.variant)}, repeat ${doc.seeds.indexOf(r.seed) + 1}`.slice(0, 100);
              return (
                <tr key={r.id} className="border-t border-slate-100 align-top">
                  <td className="py-1.5 text-slate-800">
                    <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full" style={{ background: SERIES_COLOR[doc.variants.indexOf(r.variant)] ?? "#94a3b8" }} />
                    {variantLabel(r.variant)}
                  </td>
                  <td className="py-1.5 text-slate-600">{doc.seeds.indexOf(r.seed) + 1}</td>
                  <td className="py-1.5 text-slate-700">
                    {RUN_STATUS[r.status] ?? r.status}
                    {r.error && <div className="max-w-md truncate text-xs text-red-700" title={r.error}>{r.error}</div>}
                  </td>
                  <td className="py-1.5 text-right text-slate-600">{num(r.wall_time_s) ? seconds(r.wall_time_s) : ""}</td>
                  <td className="py-1.5 text-right">
                    {r.replay && <button onClick={() => watch(r.id, label)} className="rounded-full bg-violet-600 px-3 py-0.5 text-xs text-white">▶ Watch</button>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}
