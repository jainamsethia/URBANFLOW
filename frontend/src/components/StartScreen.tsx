// The start screen: choose a map (bundled, your own imports, or a quick grid), how the
// traffic lights work and how much traffic, then start. Also imports your own data.

import { useEffect, useState } from "react";
import { api, downloads, type ImportIssue, type Meta, type ScenarioRef, type SessionInfo } from "../lib/api";
import { brain, brainChoices, town, TOWN_ORDER } from "../lib/friendly";

export const pill = (on: boolean, onColor = "bg-sky-600 text-white ring-sky-600") =>
  `rounded-full px-4 py-2 text-sm font-medium ring-1 transition ${on ? onColor : "bg-white text-slate-800 ring-slate-300 hover:bg-slate-50"}`;

export function Logo() {
  return (
    <div className="flex items-center gap-2">
      <span className="text-3xl" aria-hidden>🚦</span>
      <div>
        <div className="text-xl font-bold tracking-tight text-slate-900">UrbanFlow</div>
        <div className="-mt-0.5 text-xs text-slate-500">Traffic simulator</div>
      </div>
    </div>
  );
}

const CAR_AMOUNTS = [
  { label: "A few cars", rate: 150 },
  { label: "Some cars", rate: 300 },
  { label: "Lots of cars", rate: 600 },
];

const TRAFFIC = [
  { label: "Light", factor: 0.5, hint: "half the usual traffic" },
  { label: "Normal", factor: 1, hint: "the map's own traffic" },
  { label: "Heavy", factor: 1.5, hint: "50% more cars" },
  { label: "Very heavy", factor: 2, hint: "twice the cars" },
];

const card = "rounded-2xl bg-white p-4 shadow-sm";

export function StartScreen({
  meta,
  scenarios,
  onCreated,
  onScenariosChanged,
  onHelp,
  onEdit,
}: {
  meta: Meta;
  scenarios: ScenarioRef[];
  onCreated: (s: SessionInfo) => void;
  onScenariosChanged: () => Promise<void>;
  onHelp: () => void;
  /** Open the editor on a map (``null``: a new, empty map). */
  onEdit: (map: ScenarioRef | null) => void;
}) {
  const bundled = TOWN_ORDER.flatMap((n) => scenarios.filter((s) => s.source === "bundled" && s.name === n));
  const otherBundled = scenarios.filter((s) => s.source === "bundled" && !TOWN_ORDER.includes(s.name));
  const mine = scenarios.filter((s) => s.source === "workspace");
  const [pick, setPick] = useState(bundled.find((s) => s.name === "grid_3x3") ? "bundled:grid_3x3" : `${scenarios[0]?.source}:${scenarios[0]?.name}`);
  const [grid, setGrid] = useState({ rows: 3, cols: 3, entry_rate: 300 });
  const [controller, setController] = useState("");
  const [traffic, setTraffic] = useState(1);
  const [record, setRecord] = useState(false);
  const [guidance, setGuidance] = useState(false);
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);

  const start = async () => {
    setBusy(true);
    setError(null);
    const [source, name] = pick.split(":") as [string, string];
    const what =
      source === "generate"
        ? { generator: "grid", params: { ...grid, lanes: 2 }, label: `grid_${grid.rows}x${grid.cols}` }
        : { scenario: name, source };
    try {
      onCreated(await api.createSession({ ...what, seed, demand_scale: traffic, record, router: guidance ? "dynamic" : "shortest", ...(controller ? { controller } : {}) }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (name: string) => {
    if (!confirm(`Remove the map "${name}" from this workspace?`)) return;
    await api.deleteScenario(name).catch((e) => setError((e as Error).message));
    if (pick === `workspace:${name}`) setPick("bundled:grid_3x3");
    await onScenariosChanged();
  };

  const stepper = (key: "rows" | "cols", label: string) => (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-28 text-slate-600">{label}</span>
      <button className="h-8 w-8 rounded-full bg-slate-100 text-lg" onClick={() => setGrid({ ...grid, [key]: Math.max(1, grid[key] - 1) })} aria-label={`fewer ${label}`}>−</button>
      <span className="w-6 text-center font-semibold">{grid[key]}</span>
      <button className="h-8 w-8 rounded-full bg-slate-100 text-lg" onClick={() => setGrid({ ...grid, [key]: Math.min(8, grid[key] + 1) })} aria-label={`more ${label}`}>+</button>
    </div>
  );

  const mapCard = (s: ScenarioRef) => {
    const key = `${s.source}:${s.name}`;
    const t = s.source === "workspace" ? { icon: "🗺️", title: s.name, blurb: "Your map" } : town(s.name);
    return (
      <div key={key} className={`relative rounded-2xl bg-white shadow-sm ring-2 transition ${pick === key ? "ring-sky-500" : "ring-transparent"}`}>
        <button onClick={() => setPick(key)} aria-pressed={pick === key} className="w-full p-4 text-left">
          <div className="text-3xl" aria-hidden>{t.icon}</div>
          <div className="mt-2 font-semibold text-slate-900">{t.title}</div>
          <div className="text-sm text-slate-600">{t.blurb}</div>
        </button>
        <div className="flex gap-3 px-4 pb-3 text-xs">
          <button className="text-sky-700 underline" onClick={() => onEdit(s)} title={s.source === "workspace" ? "Change this map" : "Change a copy of this map"}>✏️ Edit</button>
          {s.source === "workspace" && (
            <>
              <a className="text-sky-700 underline" href={downloads.scenario("workspace", s.name)}>Download</a>
              <button className="text-red-700 underline" onClick={() => remove(s.name)}>Remove</button>
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-4 md:p-8">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <Logo />
        <div className="flex items-center gap-3">
          <p className="hidden text-sm text-slate-600 md:block">Pick a map, start the traffic, and see how signal timing changes everyone's trip.</p>
          <button onClick={onHelp} className={pill(false)}>❓ Help</button>
        </div>
      </header>

      <section>
        <h2 className="mb-3 text-2xl font-bold text-slate-900">1. Choose a map</h2>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          {[...bundled, ...otherBundled, ...mine].map(mapCard)}
          <div onClick={() => setPick("generate:grid")} role="button" tabIndex={0} aria-pressed={pick === "generate:grid"}
            onKeyDown={(e) => e.key === "Enter" && setPick("generate:grid")}
            className={`cursor-pointer rounded-2xl bg-white p-4 text-left shadow-sm ring-2 transition ${pick === "generate:grid" ? "ring-sky-500" : "ring-transparent"}`}>
            <div className="text-3xl" aria-hidden>🧩</div>
            <div className="mt-2 font-semibold text-slate-900">Build a grid</div>
            <div className="mb-2 text-sm text-slate-600">Choose how many streets and cars.</div>
            {pick === "generate:grid" && (
              <div className="space-y-2" onClick={(e) => e.stopPropagation()}>
                {stepper("rows", "Streets across")}
                {stepper("cols", "Streets down")}
                <div className="flex flex-wrap gap-1.5">
                  {CAR_AMOUNTS.map((a) => (
                    <button key={a.rate} onClick={() => setGrid({ ...grid, entry_rate: a.rate })} className={pill(grid.entry_rate === a.rate)}>{a.label}</button>
                  ))}
                </div>
              </div>
            )}
          </div>
          <button onClick={() => onEdit(null)} className="rounded-2xl border-2 border-dashed border-slate-300 bg-white/60 p-4 text-left hover:border-sky-400">
            <div className="text-3xl" aria-hidden>✏️</div>
            <div className="mt-2 font-semibold text-slate-900">Draw your own map</div>
            <div className="text-sm text-slate-600">Place junctions, connect them with roads, choose the lights and the traffic.</div>
          </button>
          <button onClick={() => setImporting(true)} className="rounded-2xl border-2 border-dashed border-slate-300 bg-white/60 p-4 text-left hover:border-sky-400">
            <div className="text-3xl" aria-hidden>⬆️</div>
            <div className="mt-2 font-semibold text-slate-900">Import your own map</div>
            <div className="text-sm text-slate-600">A real area from OpenStreetMap, an UrbanFlow scenario, or CityFlow files.</div>
          </button>
        </div>
      </section>

      {importing && (
        <ImportPanel
          onClose={() => setImporting(false)}
          onImported={async (name) => {
            await onScenariosChanged();
            setPick(`workspace:${name}`);
          }}
        />
      )}

      <section>
        <h2 className="mb-3 text-2xl font-bold text-slate-900">2. How should the traffic lights work?</h2>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {["", ...brainChoices(meta.controllers)].map((c) => {
            const b = c ? brain(c) : { icon: "✨", title: "As the map is set up", blurb: "Use the signal plan that comes with the map." };
            return (
              <button key={c || "default"} onClick={() => setController(c)} aria-pressed={controller === c}
                className={`rounded-2xl bg-white p-3 text-left shadow-sm ring-2 transition ${controller === c ? "ring-violet-500" : "ring-transparent"}`}>
                <span className="mr-2 text-2xl" aria-hidden>{b.icon}</span>
                <span className="font-semibold text-slate-900">{b.title}</span>
                <div className="mt-1 text-sm text-slate-600">{b.blurb}</div>
              </button>
            );
          })}
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-2xl font-bold text-slate-900">3. How much traffic?</h2>
        <div className="flex flex-wrap gap-2">
          {TRAFFIC.map((t) => (
            <button key={t.factor} onClick={() => setTraffic(t.factor)} className={pill(traffic === t.factor)} title={t.hint}>
              {t.label} <span className="opacity-70">({t.hint})</span>
            </button>
          ))}
        </div>
        <label className="mt-3 flex items-center gap-2 text-sm text-slate-700">
          <input type="checkbox" checked={guidance} onChange={(e) => setGuidance(e.target.checked)} className="h-4 w-4" />
          🧭 Drivers avoid jams (like a navigation app: routes change when a road gets slow)
        </label>
        <label className="mt-2 flex items-center gap-2 text-sm text-slate-700">
          <input type="checkbox" checked={record} onChange={(e) => setRecord(e.target.checked)} className="h-4 w-4" />
          ⏺ Record this run so I can watch it again or download it later
        </label>
      </section>

      <div className="flex flex-col items-center gap-2">
        <button disabled={busy} onClick={start}
          className="rounded-full bg-emerald-600 px-10 py-4 text-2xl font-bold text-white shadow-lg transition hover:bg-emerald-500 disabled:opacity-60">
          {busy ? "Setting up…" : "▶ Start simulation"}
        </button>
        {error && <p className="max-w-xl whitespace-pre-wrap rounded-lg bg-red-50 p-3 text-sm text-red-800">Could not start: {error}</p>}
      </div>

      <details className={`${card} text-sm text-slate-600`}>
        <summary className="cursor-pointer font-medium text-slate-700">⚙️ Advanced options</summary>
        <div className="mt-3 flex flex-wrap items-end gap-6">
          <label>
            <span className="block text-xs text-slate-500">Random seed (the same seed gives the same cars)</span>
            <input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} className="w-28 rounded-lg border border-slate-300 bg-white p-1.5" />
          </label>
          <Recordings onCreated={onCreated} />
        </div>
      </details>
    </div>
  );
}

// ---------------------------------------------------------------------------- import
async function readJson(file: File): Promise<unknown> {
  const text = await file.text();
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`"${file.name}" is not a valid JSON file.`);
  }
}

const safeName = (s: string) => s.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^[^A-Za-z0-9]+/, "").slice(0, 60) || "my_map";

function ImportPanel({ onClose, onImported }: { onClose: () => void; onImported: (name: string) => Promise<void> }) {
  const [kind, setKind] = useState<"osm" | "urbanflow" | "cityflow">("osm");
  const [name, setName] = useState("");
  const [files, setFiles] = useState<Record<string, File | null>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<{ name: string; warnings: ImportIssue[]; summary: Record<string, number> } | null>(null);

  const setFile = (key: string, f: File | null) => {
    setFiles({ ...files, [key]: f });
    if (f && !name && (key === "scenario" || key === "roadnet" || key === "osm")) setName(safeName(f.name));
  };

  const run = async (overwrite = false) => {
    setBusy(true);
    setError(null);
    try {
      let body: Record<string, unknown>;
      if (kind === "osm") {
        if (!files.osm) throw new Error("Choose the .osm file you exported from OpenStreetMap.");
        body = { name, format: "osm", osm: await files.osm.text(), overwrite };
      } else if (kind === "urbanflow") {
        if (!files.scenario) throw new Error("Choose a scenario file first.");
        body = { name, format: "urbanflow", scenario: await readJson(files.scenario), overwrite };
      } else {
        if (!files.roadnet || !files.flow) throw new Error("Choose both the road network file and the traffic (flow) file.");
        body = {
          name, format: "cityflow", overwrite,
          roadnet: await readJson(files.roadnet), flow: await readJson(files.flow),
          ...(files.config ? { config: await readJson(files.config) } : {}),
        };
      }
      const result = await api.importScenario(body);
      setDone(result);
      await onImported(result.name);
    } catch (e) {
      const msg = (e as Error).message;
      if (msg.includes("already exists") && confirm(`A map named "${name}" already exists. Replace it?`)) {
        setBusy(false);
        return run(true);
      }
      setError(msg);
    } finally {
      setBusy(false);
    }
  };

  const fileInput = (key: string, label: string, required: boolean, accept = ".json,application/json") => (
    <label className="block text-sm">
      <span className="font-medium text-slate-700">{label}{required ? "" : " (optional)"}</span>
      <input type="file" accept={accept} className="mt-1 block w-full text-sm" onChange={(e) => setFile(key, e.target.files?.[0] ?? null)} />
    </label>
  );

  return (
    <section className={`${card} space-y-3 ring-2 ring-sky-300`}>
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-900">⬆️ Import your own map</h3>
        <button onClick={onClose} className="text-slate-500 hover:text-slate-800" aria-label="Close">✕</button>
      </div>
      <div className="flex flex-wrap gap-2">
        <button className={pill(kind === "osm")} onClick={() => setKind("osm")}>🌍 OpenStreetMap</button>
        <button className={pill(kind === "urbanflow")} onClick={() => setKind("urbanflow")}>UrbanFlow scenario</button>
        <button className={pill(kind === "cityflow")} onClick={() => setKind("cityflow")}>CityFlow files</button>
      </div>
      {kind === "osm" ? (
        <>
          <ol className="list-decimal space-y-1 pl-5 text-sm text-slate-600">
            <li>Open <a className="text-sky-700 underline" href="https://www.openstreetmap.org" target="_blank" rel="noreferrer">openstreetmap.org</a> and zoom to the area you want (a few streets up to a neighbourhood works best).</li>
            <li>Click <b>Export</b> (top bar), then the blue <b>Export</b> button. Your browser saves a file called <code>map.osm</code>.</li>
            <li>Choose that file below. UrbanFlow turns the streets into roads and lanes, finds the traffic lights and adds traffic.</li>
          </ol>
          {fileInput("osm", "OpenStreetMap file (.osm)", true, ".osm,.xml,application/xml,text/xml")}
        </>
      ) : kind === "urbanflow" ? (
        <>
          <p className="text-sm text-slate-600">A <code>.json</code> scenario made with UrbanFlow (for example with <code>urbanflow generate</code> or downloaded from this app).</p>
          {fileInput("scenario", "Scenario file", true)}
        </>
      ) : (
        <>
          <p className="text-sm text-slate-600">CityFlow's road network and flow files. UrbanFlow converts them and tells you about anything it had to change.</p>
          {fileInput("roadnet", "Road network (roadnet.json)", true)}
          {fileInput("flow", "Traffic (flow.json)", true)}
          {fileInput("config", "Settings (config.json)", false)}
        </>
      )}
      <label className="block text-sm">
        <span className="font-medium text-slate-700">Name for this map</span>
        <input value={name} onChange={(e) => setName(e.target.value.replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 64))}
          placeholder="my_city" className="mt-1 block w-full rounded-lg border border-slate-300 p-1.5" />
      </label>
      <button disabled={busy || !name} onClick={() => run()} className="rounded-full bg-sky-600 px-5 py-2 font-semibold text-white disabled:opacity-50">
        {busy ? "Importing…" : "Import"}
      </button>
      {error && (
        <div className="rounded-lg bg-red-50 p-3 text-sm text-red-800">
          <div className="mb-1 font-semibold">This file could not be imported:</div>
          <pre className="max-h-60 overflow-auto whitespace-pre-wrap font-sans">{error}</pre>
        </div>
      )}
      {done && (
        <div className="rounded-lg bg-emerald-50 p-3 text-sm text-emerald-900">
          ✅ Imported <b>{done.name}</b>: {done.summary.intersections} junctions, {done.summary.roads} roads, {done.summary.flows} traffic flows. It is selected above.
          {done.warnings.length > 0 && (
            <details className="mt-2">
              <summary className="cursor-pointer">{done.warnings.length} note{done.warnings.length > 1 ? "s" : ""} about the conversion</summary>
              <ul className="mt-1 list-disc pl-5 text-xs">
                {done.warnings.map((w, i) => <li key={i}><code>{w.path}</code>: {w.message}</li>)}
              </ul>
            </details>
          )}
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------- recordings
function Recordings({ onCreated }: { onCreated: (s: SessionInfo) => void }) {
  const [replays, setReplays] = useState<{ name: string }[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.replays().then((r) => {
      setReplays(r);
      setName(r[r.length - 1]?.name ?? "");
    }).catch(() => {});
  }, []);
  const open = async () => {
    setError(null);
    try {
      onCreated(await api.createSession({ replay: name }));
    } catch (e) {
      setError((e as Error).message);
    }
  };
  return (
    <div className="flex flex-wrap items-end gap-2">
      <label>
        <span className="block text-xs text-slate-500">Recordings</span>
        <select className="max-w-64 rounded-lg border border-slate-300 bg-white p-1.5" value={name} onChange={(e) => setName(e.target.value)}>
          {replays.length === 0 && <option value="">no recordings yet</option>}
          {replays.map((r) => <option key={r.name}>{r.name}</option>)}
        </select>
      </label>
      <button disabled={!name} onClick={open} className="rounded-full bg-violet-600 px-4 py-1.5 text-white disabled:opacity-40">▶ Watch</button>
      {name && <a className="text-sm text-sky-700 underline" href={downloads.replay(name)}>Download</a>}
      {error && <span className="text-xs text-red-700">{error}</span>}
    </div>
  );
}
