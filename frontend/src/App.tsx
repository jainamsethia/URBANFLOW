import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { MapCanvas, type VehicleColoring } from "./components/MapCanvas";
import { ComparePanel, Inspector, MetricsPanel, clock, fmt } from "./components/Panels";
import { api, type Geometry, type Meta, type ScenarioRef, type SessionInfo } from "./lib/api";
import { SessionController, type Snapshot } from "./lib/session";

const SPEEDS: (number | null)[] = [1, 5, 10, 25, 50, 100, 250, 1000, null];
const REPLAY_SPEEDS: number[] = [-250, -50, -10, 1, 5, 10, 25, 50, 100, 250, 1000];

function useSnapshot(ctl: SessionController | null): Snapshot | null {
  const sub = useCallback((fn: () => void) => (ctl ? ctl.subscribe(fn) : () => {}), [ctl]);
  return useSyncExternalStore(sub, () => (ctl ? ctl.snapshot : null));
}

function NewSession({
  meta,
  scenarios,
  onCreated,
}: {
  meta: Meta;
  scenarios: ScenarioRef[];
  onCreated: (s: SessionInfo) => void;
}) {
  const [scenario, setScenario] = useState(`bundled:${scenarios.find((s) => s.name === "grid_3x3")?.name ?? scenarios[0]?.name ?? ""}`);
  const [controller, setController] = useState("");
  const [seed, setSeed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const create = async () => {
    setBusy(true);
    setError(null);
    const [source, name] = scenario.split(":") as ["bundled" | "workspace", string];
    try {
      onCreated(
        await api.createSession({ scenario: name, source, seed, ...(controller ? { controller } : {}) }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex items-end gap-2 text-sm">
      <label>
        <span className="block text-[11px] text-slate-400">Scenario</span>
        <select className="rounded bg-slate-800 p-1" value={scenario} onChange={(e) => setScenario(e.target.value)}>
          {scenarios.map((s) => (
            <option key={`${s.source}:${s.name}`} value={`${s.source}:${s.name}`}>
              {s.name}
              {s.source === "workspace" ? " (workspace)" : ""}
            </option>
          ))}
        </select>
      </label>
      <label>
        <span className="block text-[11px] text-slate-400">Signals</span>
        <select className="rounded bg-slate-800 p-1" value={controller} onChange={(e) => setController(e.target.value)}>
          <option value="">scenario default</option>
          {meta.controllers.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      </label>
      <label>
        <span className="block text-[11px] text-slate-400">Seed</span>
        <input type="number" min={0} className="w-16 rounded bg-slate-800 p-1" value={seed} onChange={(e) => setSeed(+e.target.value)} />
      </label>
      <button disabled={busy} onClick={create} className="rounded bg-sky-600 px-3 py-1 font-medium hover:bg-sky-500 disabled:opacity-50">
        {busy ? "Starting…" : "New session"}
      </button>
      {error && <span className="max-w-xs truncate text-xs text-red-400" title={error}>{error}</span>}
    </div>
  );
}

function OpenReplay({ onCreated }: { onCreated: (s: SessionInfo) => void }) {
  const [replays, setReplays] = useState<{ name: string }[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const refresh = () =>
    api.replays().then((r) => {
      setReplays(r);
      setName((cur) => cur || r[r.length - 1]?.name || "");
    });
  useEffect(() => {
    refresh().catch(() => {});
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
    <div className="flex items-end gap-2 text-sm">
      <label>
        <span className="block text-[11px] text-slate-400">Replays</span>
        <select className="max-w-48 rounded bg-slate-800 p-1" value={name} onFocus={() => refresh()} onChange={(e) => setName(e.target.value)}>
          {replays.length === 0 && <option value="">none recorded</option>}
          {replays.map((r) => (
            <option key={r.name}>{r.name}</option>
          ))}
        </select>
      </label>
      <button disabled={!name} onClick={open} className="rounded bg-violet-700 px-3 py-1 font-medium hover:bg-violet-600 disabled:opacity-40">
        Open replay
      </button>
      {error && <span className="max-w-xs truncate text-xs text-red-400" title={error}>{error}</span>}
    </div>
  );
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioRef[]>([]);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [ctl, setCtl] = useState<SessionController | null>(null);
  const [tab, setTab] = useState<"live" | "compare">("live");
  const [coloring, setColoring] = useState<VehicleColoring>("type");
  const [laneMetric, setLaneMetric] = useState(0);
  const [selected, setSelected] = useState<{ kind: string; id: string }>({ kind: "none", id: "" });
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
    send({ type: "select", kind, id });
  };
  const playing = status?.state === "playing";
  const isReplay = status?.kind === "replay";
  const signalised = Object.keys(status?.controllers ?? {});
  const liveController = signalised.length ? status!.controllers[signalised[0]!] : "";

  if (fatal) return <div className="p-6 text-red-300">Cannot reach the UrbanFlow server: {fatal}</div>;
  if (!meta) return <div className="p-6 text-slate-400">Connecting…</div>;

  return (
    <div className="flex h-screen flex-col">
      <header className="flex flex-wrap items-center gap-4 border-b border-slate-800 bg-slate-900 px-4 py-2">
        <div className="flex items-baseline gap-2">
          <span className="text-lg font-semibold tracking-tight text-sky-400">UrbanFlow</span>
          <span className="text-xs text-slate-500">v{meta.version} · {meta.project}</span>
        </div>
        <NewSession meta={meta} scenarios={scenarios} onCreated={setSession} />
        <OpenReplay onCreated={setSession} />
        {session && (
          <div className="ml-auto flex items-center gap-3 text-sm">
            <span className={`rounded px-2 py-0.5 text-xs ${playing ? "bg-emerald-800" : status?.state === "ended" ? "bg-slate-700" : status?.state === "error" ? "bg-red-800" : "bg-amber-800"}`}>
              {status?.state ?? "connecting"}
            </span>
            <span className="font-mono text-slate-300">
              t={clock(status?.time ?? 0)} · step {status?.step ?? 0} · {status?.vehicles ?? 0} veh
            </span>
          </div>
        )}
      </header>

      {session && (
        <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 bg-slate-900/60 px-4 py-1.5 text-sm">
          <span className="mr-2 text-slate-300">{session.label}</span>
          <button onClick={() => send({ type: playing ? "pause" : "play" })} className="w-20 rounded bg-sky-700 py-1 hover:bg-sky-600" disabled={!snap?.connected}>
            {playing ? "❚❚ Pause" : "▶ Play"}
          </button>
          {isReplay && (
            <button onClick={() => send({ type: "step", n: -1 })} className="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700" title="Step back">-1</button>
          )}
          <button onClick={() => send({ type: "step", n: 1 })} className="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700">+1</button>
          <button onClick={() => send({ type: "step", n: 60 })} className="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700">+60</button>
          <button onClick={() => confirm("Reset the simulation?") && send({ type: "reset" })} className="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700">Reset</button>
          <label className="ml-2 flex items-center gap-1 text-xs text-slate-400">
            Speed
            <select className="rounded bg-slate-800 p-1 text-slate-100" value={String(status?.steps_per_second ?? "max")} onChange={(e) => send({ type: "set_speed", steps_per_second: e.target.value === "max" ? null : +e.target.value })}>
              {(isReplay ? REPLAY_SPEEDS : SPEEDS).map((s) => (
                <option key={String(s)} value={s == null ? "max" : String(s)}>
                  {s == null ? "max" : s < 0 ? `rewind ${-s}/s` : `${s} steps/s`}
                </option>
              ))}
            </select>
          </label>
          {signalised.length > 0 && !isReplay && (
            <label className="ml-2 flex items-center gap-1 text-xs text-slate-400">
              Signals
              <select className="rounded bg-slate-800 p-1 text-slate-100" value={liveController} onChange={(e) => send({ type: "set_controller", controller: e.target.value })}>
                {[...new Set([liveController, ...meta.controllers])].filter(Boolean).map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
          )}
          <label className="ml-2 flex items-center gap-1 text-xs text-slate-400">
            View
            <select
              className="rounded bg-slate-800 p-1 text-slate-100"
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
              }}
            >
              <option value="veh:type">Vehicles by type</option>
              <option value="veh:speed">Vehicles by speed</option>
              {!isReplay && meta.lane_metrics.map((m) => (
                <option key={m.code} value={`lane:${m.code}`}>
                  Lanes: {m.label}
                  {m.unit ? ` (${m.unit})` : ""}
                </option>
              ))}
            </select>
          </label>
          {isReplay && status && (
            <input
              type="range"
              aria-label="Replay timeline"
              className="ml-2 min-w-48 flex-1 accent-violet-500"
              min={status.first_step ?? 0}
              max={status.last_step ?? 0}
              value={status.step}
              onChange={(e) => send({ type: "seek", step: +e.target.value })}
            />
          )}
          {snap?.error && <span className="ml-2 truncate text-xs text-red-400" title={snap.error}>{snap.error}</span>}
        </div>
      )}

      <main className="flex min-h-0 flex-1">
        <section className="relative min-w-0 flex-1">
          {session && geometry && ctl ? (
            <MapCanvas
              geometry={geometry}
              controller={ctl}
              coloring={coloring}
              laneMetric={laneInfo}
              selectedVehicle={selected.kind === "vehicle" ? selected.id : null}
              selectedIntersection={selected.kind === "intersection" ? selected.id : null}
              onSelect={select}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-slate-500">
              {session ? "Loading network…" : "Pick a scenario and start a session."}
            </div>
          )}
          {laneInfo && (
            <div className="absolute bottom-3 left-3 rounded bg-slate-900/85 p-2 text-[11px] text-slate-300">
              <div className="mb-1">{laneInfo.label} {laneInfo.unit && `(${laneInfo.unit})`}</div>
              <div className="h-2 w-40 rounded" style={{ background: "linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)" }} />
              <div className="flex justify-between text-slate-500">
                <span>{fmt(laneInfo.domain?.[0] ?? 0)}</span>
                <span>{laneInfo.domain ? fmt(laneInfo.domain[1]) : "max"}</span>
              </div>
            </div>
          )}
        </section>

        <aside className="flex w-80 flex-col border-l border-slate-800 bg-slate-950">
          <div className="flex border-b border-slate-800 text-sm">
            {(["live", "compare"] as const).map((t) => (
              <button key={t} onClick={() => setTab(t)} className={`flex-1 py-2 ${tab === t ? "border-b-2 border-sky-500 text-sky-300" : "text-slate-400"}`}>
                {t === "live" ? "Live" : "Compare controllers"}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3">
            {tab === "live" ? (
              snap && ctl ? (
                <>
                  <MetricsPanel snap={snap} />
                  <div className="rounded-lg border border-slate-800 p-2">
                    <Inspector selection={snap.selection && selected.kind !== "none" ? snap.selection : null} controller={ctl} />
                  </div>
                </>
              ) : (
                <p className="text-sm text-slate-500">No session yet.</p>
              )
            ) : (
              <ComparePanel meta={meta} scenarios={scenarios} />
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}
