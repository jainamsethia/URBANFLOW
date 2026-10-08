import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { MapCanvas, type VehicleColoring } from "./components/MapCanvas";
import { MoodCard, RacePanel, SelectionCard, StatTiles, TechnicalDetails, fmt } from "./components/Panels";
import { api, type Geometry, type Meta, type ScenarioRef, type SessionInfo } from "./lib/api";
import { brain, brainChoices, clock, SPEEDS, town, TOWN_ORDER, VEHICLE } from "./lib/friendly";
import { SessionController, remainingRoute, type Snapshot } from "./lib/session";

function useSnapshot(ctl: SessionController | null): Snapshot | null {
  const sub = useCallback((fn: () => void) => (ctl ? ctl.subscribe(fn) : () => {}), [ctl]);
  return useSyncExternalStore(sub, () => (ctl ? ctl.snapshot : null));
}

const pill = (on: boolean, onColor = "bg-sky-600 text-white ring-sky-600") =>
  `rounded-full px-4 py-2 text-sm font-medium ring-1 transition ${on ? onColor : "bg-white text-slate-800 ring-slate-300 hover:bg-slate-50"}`;

function Logo() {
  return (
    <div className="flex items-center gap-2">
      <span className="text-3xl" aria-hidden>🚦</span>
      <div>
        <div className="text-xl font-bold tracking-tight text-slate-900">UrbanFlow</div>
        <div className="-mt-0.5 text-xs text-slate-500">Traffic playground</div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------- start screen
const CAR_AMOUNTS = [
  { label: "A few cars", rate: 150 },
  { label: "Some cars", rate: 300 },
  { label: "Lots of cars", rate: 600 },
];

function StartScreen({ meta, scenarios, onCreated }: { meta: Meta; scenarios: ScenarioRef[]; onCreated: (s: SessionInfo) => void }) {
  const ordered = [
    ...TOWN_ORDER.flatMap((n) => scenarios.filter((s) => s.source === "bundled" && s.name === n)),
    ...scenarios.filter((s) => !(s.source === "bundled" && TOWN_ORDER.includes(s.name))),
  ];
  const [pick, setPick] = useState(ordered.find((s) => s.name === "grid_3x3") ? "bundled:grid_3x3" : `${ordered[0]?.source}:${ordered[0]?.name}`);
  const [grid, setGrid] = useState({ rows: 3, cols: 3, entry_rate: 300 });
  const [controller, setController] = useState("");
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = async () => {
    setBusy(true);
    setError(null);
    const [source, name] = pick.split(":") as [string, string];
    const what =
      source === "generate"
        ? { generator: "grid", params: { ...grid, lanes: 2 }, label: `my_town_${grid.rows}x${grid.cols}` }
        : { scenario: name, source };
    try {
      onCreated(await api.createSession({ ...what, seed, ...(controller ? { controller } : {}) }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const stepper = (key: "rows" | "cols", label: string) => (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-24 text-slate-600">{label}</span>
      <button className="h-8 w-8 rounded-full bg-slate-100 text-lg" onClick={() => setGrid({ ...grid, [key]: Math.max(1, grid[key] - 1) })} aria-label={`fewer ${label}`}>−</button>
      <span className="w-6 text-center font-semibold">{grid[key]}</span>
      <button className="h-8 w-8 rounded-full bg-slate-100 text-lg" onClick={() => setGrid({ ...grid, [key]: Math.min(6, grid[key] + 1) })} aria-label={`more ${label}`}>+</button>
    </div>
  );

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-4 md:p-8">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <Logo />
        <p className="text-sm text-slate-600">Build a town, start the cars, and watch how traffic lights keep everyone moving.</p>
      </header>

      <section>
        <h2 className="mb-3 text-2xl font-bold text-slate-900">1. Pick a town</h2>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          {ordered.map((s) => {
            const key = `${s.source}:${s.name}`;
            const t = town(s.name);
            return (
              <button key={key} onClick={() => setPick(key)} aria-pressed={pick === key}
                className={`rounded-2xl bg-white p-4 text-left shadow-sm ring-2 transition hover:-translate-y-0.5 ${pick === key ? "ring-sky-500" : "ring-transparent"}`}>
                <div className="text-4xl" aria-hidden>{t.icon}</div>
                <div className="mt-2 text-lg font-semibold text-slate-900">{t.title}</div>
                <div className="text-sm text-slate-600">{t.blurb}</div>
              </button>
            );
          })}
          <div onClick={() => setPick("generate:grid")} role="button" tabIndex={0} aria-pressed={pick === "generate:grid"}
            onKeyDown={(e) => e.key === "Enter" && setPick("generate:grid")}
            className={`cursor-pointer rounded-2xl bg-white p-4 text-left shadow-sm ring-2 transition ${pick === "generate:grid" ? "ring-sky-500" : "ring-transparent"}`}>
            <div className="text-4xl" aria-hidden>🧩</div>
            <div className="mt-2 text-lg font-semibold text-slate-900">Build my own town</div>
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
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-2xl font-bold text-slate-900">2. How should the traffic lights think?</h2>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {["", ...brainChoices(meta.controllers)].map((c) => {
            const b = c ? brain(c) : { icon: "✨", title: "The town's own lights", blurb: "Use the lights this town comes with." };
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

      <div className="flex flex-col items-center gap-2">
        <button disabled={busy} onClick={start}
          className="rounded-full bg-emerald-600 px-10 py-4 text-2xl font-bold text-white shadow-lg transition hover:bg-emerald-500 disabled:opacity-60">
          {busy ? "Building the town…" : "▶ Let's go!"}
        </button>
        {error && <p className="max-w-xl whitespace-pre-wrap rounded-lg bg-red-50 p-2 text-sm text-red-800">Oops! {error}</p>}
      </div>

      <details className="rounded-2xl bg-white p-4 text-sm text-slate-600 shadow-sm">
        <summary className="cursor-pointer font-medium text-slate-700">🔧 For grown-ups</summary>
        <div className="mt-3 flex flex-wrap items-end gap-4">
          <label>
            <span className="block text-xs text-slate-500">Random seed (same seed = same cars)</span>
            <input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} className="w-28 rounded-lg border border-slate-300 bg-white p-1.5" />
          </label>
          <OpenReplay onCreated={onCreated} />
        </div>
      </details>
    </div>
  );
}

function OpenReplay({ onCreated }: { onCreated: (s: SessionInfo) => void }) {
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
    <div className="flex items-end gap-2">
      <label>
        <span className="block text-xs text-slate-500">Watch a recording</span>
        <select className="max-w-56 rounded-lg border border-slate-300 bg-white p-1.5" value={name} onChange={(e) => setName(e.target.value)}>
          {replays.length === 0 && <option value="">no recordings yet</option>}
          {replays.map((r) => <option key={r.name}>{r.name}</option>)}
        </select>
      </label>
      <button disabled={!name} onClick={open} className="rounded-full bg-violet-600 px-4 py-1.5 text-white disabled:opacity-40">Open</button>
      {error && <span className="text-xs text-red-700">{error}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------- session view
function Legend() {
  const swatch = (color: string) => <span className="inline-block h-2.5 w-4 rounded-sm ring-1 ring-white" style={{ background: color }} />;
  return (
    <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap gap-x-3 gap-y-1 rounded-xl bg-white/90 px-3 py-2 text-xs text-slate-700 shadow">
      <span>🟢 Go</span>
      <span>🟡 Slow down</span>
      <span>🔴 Stop</span>
      <span className="mx-1 text-slate-300">|</span>
      {(["car", "bus", "truck", "emergency"] as const).map((k) => (
        <span key={k} className="flex items-center gap-1">
          {swatch(VEHICLE_COLORS[k]!)} {VEHICLE[k]!.name}
        </span>
      ))}
    </div>
  );
}

const VEHICLE_COLORS: Record<string, string> = { car: "#2a78d6", bus: "#eda100", truck: "#4a3aa7", emergency: "#e34948" };

const STATE_PILL: Record<string, [string, string]> = {
  playing: ["▶ Running", "bg-emerald-100 text-emerald-800"],
  paused: ["⏸ Paused", "bg-amber-100 text-amber-800"],
  ended: ["🏁 Finished!", "bg-sky-100 text-sky-800"],
  error: ["⚠️ Something went wrong", "bg-red-100 text-red-800"],
};

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioRef[]>([]);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [ctl, setCtl] = useState<SessionController | null>(null);
  const [tab, setTab] = useState<"traffic" | "race" | "tools">("traffic");
  const [coloring, setColoring] = useState<VehicleColoring>("type");
  const [laneMetric, setLaneMetric] = useState(0);
  const [selected, setSelected] = useState<{ kind: string; id: string }>({ kind: "none", id: "" });
  const [hint, setHint] = useState(true);
  const [fatal, setFatal] = useState<string | null>(null);
  const snap = useSnapshot(ctl);

  useEffect(() => {
    Promise.all([api.meta(), api.scenarios(), api.sessions()])
      .then(([m, s, existing]) => {
        setMeta(m);
        setScenarios(s);
        const wanted = new URLSearchParams(location.search).get("replay");
        if (wanted) {
          history.replaceState(null, "", location.pathname);
          api.createSession({ replay: wanted }).then(setSession).catch((e) => setFatal(String(e)));
        } else if (existing.length) setSession(existing[existing.length - 1]!);
      })
      .catch((e) => setFatal(String(e)));
  }, []);

  useEffect(() => {
    if (!session) return;
    let cancelled = false;
    const controller = new SessionController(session.id);
    setCtl(controller);
    setGeometry(null);
    setSelected({ kind: "none", id: "" });
    api.geometry(session.id).then((g) => !cancelled && setGeometry(g));
    api.summary(session.id).then((s) => !cancelled && controller.seedHistory(s.history));
    return () => {
      cancelled = true;
      controller.dispose();
    };
  }, [session?.id]);

  const laneInfo = useMemo(() => meta?.lane_metrics.find((m) => m.code === laneMetric) ?? null, [meta, laneMetric]);
  const status = snap?.status;
  const send = (m: Record<string, unknown>) => ctl?.send(m);
  const select = (kind: "vehicle" | "intersection" | "none", id: string) => {
    setSelected({ kind, id });
    setHint(false);
    send({ type: "select", kind, id });
  };
  const playing = status?.state === "playing";
  const isReplay = status?.kind === "replay";
  const signalised = Object.keys(status?.controllers ?? {});
  const liveController = signalised.length ? status!.controllers[signalised[0]!]! : "";
  const goHome = () => {
    if (session) api.deleteSession(session.id).catch(() => {});
    setSession(null);
    setCtl(null);
    setGeometry(null);
  };

  if (fatal) return <div className="p-8 text-lg text-red-700">😕 Oops! I can't reach the traffic computer. Is the UrbanFlow server running? ({fatal})</div>;
  if (!meta) return <div className="p-8 text-lg text-slate-500">🚦 Getting the town ready…</div>;
  if (!session) return <StartScreen meta={meta} scenarios={scenarios} onCreated={setSession} />;

  const t = town(session.scenario || session.label);
  const [stateText, stateClass] = STATE_PILL[status?.state ?? ""] ?? ["Connecting…", "bg-slate-100 text-slate-700"];
  const selection = snap?.selection && selected.kind !== "none" ? snap.selection : null;

  return (
    <div className="flex min-h-screen flex-col md:h-screen">
      <header className="flex flex-wrap items-center gap-x-6 gap-y-2 bg-white px-4 py-3 shadow-sm">
        <Logo />
        <div className="flex items-center gap-2 text-lg font-semibold text-slate-800">
          <span aria-hidden>{t.icon}</span> {t.title}
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-3 text-sm">
          <span className={`rounded-full px-3 py-1 font-medium ${stateClass}`}>{stateText}</span>
          <span className="text-slate-700">⏰ <b className="text-lg">{clock(status?.time ?? 0)}</b></span>
          <span className="text-slate-700">🚗 <b className="text-lg">{fmt(status?.vehicles ?? 0, 0)}</b> cars</span>
        </div>
      </header>

      <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 bg-slate-50 px-4 py-2">
        <button onClick={() => send({ type: playing ? "pause" : "play" })} disabled={!snap?.connected}
          className={`rounded-full px-6 py-2.5 text-lg font-bold text-white shadow transition disabled:opacity-50 ${playing ? "bg-amber-500 hover:bg-amber-600" : "bg-emerald-600 hover:bg-emerald-500"}`}>
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>
        <div className="flex flex-wrap gap-1" role="group" aria-label="Speed">
          {SPEEDS.map((s) => (
            <button key={s.label} onClick={() => send({ type: "set_speed", steps_per_second: s.sps })}
              className={pill((status?.steps_per_second ?? null) === s.sps)} title={s.sps ? `${s.sps} seconds of traffic per real second` : "As fast as the computer can"}>
              {s.icon} {s.label}
            </button>
          ))}
        </div>
        {signalised.length > 0 && !isReplay && (
          <label className="flex items-center gap-2 text-sm text-slate-700">
            <span>Lights' brain</span>
            <select className="rounded-full border border-slate-300 bg-white px-3 py-2" value={liveController}
              onChange={(e) => send({ type: "set_controller", controller: e.target.value })}>
              {[...new Set([liveController, ...brainChoices(meta.controllers)])].filter(Boolean).map((c) => (
                <option key={c} value={c}>{brain(c).icon} {brain(c).title}</option>
              ))}
            </select>
          </label>
        )}
        {isReplay && status && (
          <input type="range" aria-label="Recording timeline" className="min-w-48 flex-1 accent-violet-600"
            min={status.first_step ?? 0} max={status.last_step ?? 0} value={status.step}
            onChange={(e) => send({ type: "seek", step: +e.target.value })} />
        )}
        <div className="ml-auto flex gap-2">
          {!isReplay && (
            <button onClick={() => confirm("Start this town again from the beginning?") && send({ type: "reset" })} className={pill(false)}>🔄 Start again</button>
          )}
          <button onClick={goHome} className={pill(false)}>🏠 Pick another town</button>
        </div>
      </div>

      <main className="flex min-h-0 flex-1 flex-col md:flex-row">
        <section className="relative h-[60vh] min-h-80 min-w-0 bg-[#cfe8c4] md:h-auto md:flex-1">
          {geometry && ctl ? (
            <MapCanvas
              geometry={geometry}
              controller={ctl}
              coloring={coloring}
              laneMetric={laneInfo}
              selectedVehicle={selected.kind === "vehicle" ? selected.id : null}
              selectedIntersection={selected.kind === "intersection" ? selected.id : null}
              route={snap?.selection?.id === selected.id ? remainingRoute(snap.selection) : null}
              onSelect={select}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-lg text-slate-600">Building the town…</div>
          )}
          {hint && geometry && (
            <button onClick={() => setHint(false)} className="absolute left-3 top-3 max-w-xs rounded-xl bg-white/95 px-3 py-2 text-left text-sm text-slate-700 shadow">
              👆 <b>Click a car</b> to follow it, or <b>click a traffic light</b> to see who may go. Scroll to zoom, drag to move.
              <span className="ml-1 text-slate-400">✕</span>
            </button>
          )}
          {geometry && <Legend />}
        </section>

        <aside className="flex w-full flex-col border-t border-slate-200 bg-slate-100 md:w-96 md:border-l md:border-t-0">
          <div className="flex gap-1 p-2" role="tablist">
            {([["traffic", "📊 Traffic"], ["race", "🏁 Race"], ["tools", "🔧 Grown-ups"]] as const).map(([k, label]) => (
              <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}
                className={`flex-1 rounded-full py-2 text-sm font-medium ${tab === k ? "bg-white text-slate-900 shadow" : "text-slate-600 hover:bg-white/60"}`}>
                {label}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3 pt-1">
            {tab === "traffic" &&
              (snap && ctl && geometry ? (
                <>
                  <MoodCard snap={snap} />
                  <SelectionCard selection={selection} geometry={geometry} controller={ctl} />
                  <StatTiles snap={snap} />
                </>
              ) : (
                <p className="text-slate-500">Connecting…</p>
              ))}
            {tab === "race" && <RacePanel meta={meta} scenarios={scenarios} />}
            {tab === "tools" && snap && (
              <>
                <div className="space-y-3 rounded-2xl bg-white p-4 text-sm shadow-sm">
                  <div className="font-semibold text-slate-900">Step through time</div>
                  <div className="flex gap-2">
                    {isReplay && <button onClick={() => send({ type: "step", n: -1 })} className={pill(false)}>−1 s</button>}
                    <button onClick={() => send({ type: "step", n: 1 })} className={pill(false)}>+1 s</button>
                    <button onClick={() => send({ type: "step", n: 60 })} className={pill(false)}>+1 min</button>
                  </div>
                  <label className="block">
                    <span className="block font-semibold text-slate-900">Colour the map by</span>
                    <select className="mt-1 w-full rounded-lg border border-slate-300 bg-white p-1.5"
                      value={laneMetric ? `lane:${laneMetric}` : `veh:${coloring}`}
                      onChange={(e) => {
                        const [k, v] = e.target.value.split(":");
                        if (k === "veh") {
                          setColoring(v as VehicleColoring);
                          setLaneMetric(0);
                          send({ type: "subscribe", lane_metric: 0 });
                        } else {
                          setLaneMetric(+v!);
                          send({ type: "subscribe", lane_metric: +v! });
                        }
                      }}>
                      <option value="veh:type">Vehicles by type</option>
                      <option value="veh:speed">Vehicles by speed</option>
                      {!isReplay && meta.lane_metrics.map((m) => (
                        <option key={m.code} value={`lane:${m.code}`}>Lanes: {m.label}{m.unit ? ` (${m.unit})` : ""}</option>
                      ))}
                    </select>
                  </label>
                  {snap.error && <p className="text-red-700">{snap.error}</p>}
                </div>
                <TechnicalDetails snap={snap} selection={selection} />
                <p className="px-1 text-xs text-slate-500">
                  UrbanFlow v{meta.version} · {meta.project} · session {session.id} · {VEHICLE.car!.icon} {fmt(status?.vehicles ?? 0, 0)} vehicles
                </p>
              </>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}
