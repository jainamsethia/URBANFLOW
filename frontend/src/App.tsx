import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { BLANK_MAP, Editor, type Draft } from "./components/Editor";
import { Help } from "./components/Help";
import { MapCanvas, type Pick, type VehicleColoring } from "./components/MapCanvas";
import { EventLog, MoodCard, RacePanel, SelectionCard, StatTiles, TechnicalDetails, fmt } from "./components/Panels";
import { Logo, pill, StartScreen } from "./components/StartScreen";
import { api, downloads, type Geometry, type Meta, type ScenarioRef, type SessionInfo } from "./lib/api";
import { brain, brainChoices, clock, LANE_VIEWS, SPEEDS, town, VEHICLE } from "./lib/friendly";
import { SessionController, remainingRoute, type Snapshot } from "./lib/session";
import { SIDE_COLOR, VCLASS_COLOR } from "./lib/colors";

function useSnapshot(ctl: SessionController | null): Snapshot | null {
  const sub = useCallback((fn: () => void) => (ctl ? ctl.subscribe(fn) : () => {}), [ctl]);
  return useSyncExternalStore(sub, () => (ctl ? ctl.snapshot : null));
}


function Legend({ busStops, coloring, lanes }: { busStops: boolean; coloring: VehicleColoring; lanes: string | null }) {
  const swatch = (color: string) => <span className="inline-block h-2.5 w-4 rounded-sm ring-1 ring-white" style={{ background: color }} />;
  return (
    <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap gap-x-3 gap-y-1 rounded-xl bg-white/90 px-3 py-2 text-xs text-slate-700 shadow">
      <span>🟢 Go</span>
      <span>🟡 Slow down</span>
      <span>🔴 Stop</span>
      <span className="mx-1 text-slate-300">|</span>
      {coloring === "type" &&
        (["car", "bus", "truck", "emergency"] as const).map((k) => (
          <span key={k} className="flex items-center gap-1">
            {swatch(VCLASS_COLOR[k]!)} {VEHICLE[k]!.name}
          </span>
        ))}
      {coloring === "speed" && <span>Cars: 🟥 stopped → 🟨 slow → 🟩 fast</span>}
      {coloring === "destination" &&
        Object.entries(SIDE_COLOR).map(([side, color]) => (
          <span key={side} className="flex items-center gap-1">
            {swatch(color)} going {side}
          </span>
        ))}
      {coloring === "signals" && <span>Cars greyed out so the traffic lights stand out</span>}
      {lanes && <span>Roads coloured by {lanes}: dark = low, yellow = high</span>}
      {busStops && <span>🚏 Bus stop</span>}
    </div>
  );
}

const STATE_PILL: Record<string, [string, string]> = {
  playing: ["▶ Running", "bg-emerald-100 text-emerald-800"],
  paused: ["⏸ Paused", "bg-amber-100 text-amber-800"],
  ended: ["🏁 Finished", "bg-sky-100 text-sky-800"],
  error: ["⚠️ Stopped by an error", "bg-red-100 text-red-800"],
};

function Downloads({ session, lastRecording, isReplay }: { session: SessionInfo; lastRecording: string | null; isReplay: boolean }) {
  const link = "block rounded-lg px-3 py-2 text-sm text-slate-800 hover:bg-slate-100";
  return (
    <details className="relative">
      <summary className={`${pill(false)} cursor-pointer list-none`}>⬇ Download</summary>
      <div className="absolute right-0 z-20 mt-2 w-72 rounded-xl bg-white p-2 shadow-lg ring-1 ring-slate-200">
        {!isReplay && (
          <>
            <a className={link} href={downloads.table(session.id, "trips")}>🧾 Every finished trip <span className="text-slate-500">(CSV)</span></a>
            <a className={link} href={downloads.table(session.id, "timeseries")}>📈 The network every 10 seconds <span className="text-slate-500">(CSV)</span></a>
            <a className={link} href={downloads.table(session.id, "intersections")}>🚦 Each junction every 10 seconds <span className="text-slate-500">(CSV)</span></a>
          </>
        )}
        <a className={link} href={downloads.sessionScenario(session.id)}>🗺️ This map <span className="text-slate-500">(scenario JSON)</span></a>
        {lastRecording && <a className={link} href={downloads.replay(lastRecording)}>⏺ Last recording <span className="text-slate-500">(.ufr)</span></a>}
        <p className="px-3 pb-1 pt-2 text-xs text-slate-500">Tables open in Excel, Google Sheets or Python.</p>
      </div>
    </details>
  );
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioRef[]>([]);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [ctl, setCtl] = useState<SessionController | null>(null);
  const [tab, setTab] = useState<"traffic" | "compare" | "advanced">("traffic");
  const [coloring, setColoring] = useState<VehicleColoring>("type");
  const [laneMetric, setLaneMetric] = useState(0);
  const [selected, setSelected] = useState<{ kind: string; id: string }>({ kind: "none", id: "" });
  const [hint, setHint] = useState(true);
  const [help, setHelp] = useState(false);
  const [fatal, setFatal] = useState<string | null>(null);
  const [editing, setEditing] = useState<{ draft: Draft; name: string } | null>(null);
  const snap = useSnapshot(ctl);

  const refreshScenarios = useCallback(() => api.scenarios().then(setScenarios), []);

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

  const status = snap?.status;
  const playing = status?.state === "playing";
  const send = useCallback((m: Record<string, unknown>) => ctl?.send(m), [ctl]);

  // keyboard: Space plays/pauses, Esc closes help (F is handled by the map)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) return;
      if (e.key === "Escape") setHelp(false);
      if (e.key === " " && ctl) {
        e.preventDefault();
        send({ type: playing ? "pause" : "play" });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [ctl, playing, send]);

  const laneInfo = useMemo(() => meta?.lane_metrics.find((m) => m.code === laneMetric) ?? null, [meta, laneMetric]);
  const select = (kind: Pick, id: string) => {
    setSelected({ kind, id });
    setHint(false);
    send({ type: "select", kind, id });
  };
  const isReplay = status?.kind === "replay";
  const signalised = Object.keys(status?.controllers ?? {});
  const liveController = signalised.length ? status!.controllers[signalised[0]!]! : "";
  const goHome = () => {
    ctl?.dispose(); // close the live connection before the server ends the session
    if (session) api.deleteSession(session.id).catch(() => {});
    setSession(null);
    setCtl(null);
    setGeometry(null);
    refreshScenarios().catch(() => {});
  };

  const helpPanel = help && meta ? <Help controllers={meta.controllers} onClose={() => setHelp(false)} /> : null;
  if (fatal) return <div className="p-8 text-lg text-red-700">Cannot reach the UrbanFlow server. Is it running? ({fatal})</div>;
  if (!meta) return <div className="p-8 text-lg text-slate-500">🚦 Loading…</div>;
  if (editing)
    return (
      <Editor
        meta={meta}
        initial={editing.draft}
        initialName={editing.name}
        onExit={() => {
          setEditing(null);
          refreshScenarios().catch(() => {});
        }}
        onRun={(name) => {
          api.createSession({ scenario: name, source: "workspace" })
            .then((s) => {
              setEditing(null);
              setSession(s);
            })
            .catch((e) => setFatal(String(e)));
        }}
      />
    );
  const openEditor = (map: ScenarioRef | null) => {
    if (!map) return setEditing({ draft: BLANK_MAP, name: "my-map" });
    api.scenarioDoc(map.source, map.name)
      .then((doc) => setEditing({ draft: doc as unknown as Draft, name: map.source === "workspace" ? map.name : `${map.name}-copy` }))
      .catch((e) => setFatal(String(e)));
  };
  if (!session)
    return (
      <>
        <StartScreen meta={meta} scenarios={scenarios} onCreated={setSession} onScenariosChanged={refreshScenarios} onHelp={() => setHelp(true)} onEdit={openEditor} />
        {helpPanel}
      </>
    );

  const t = scenarios.some((s) => s.source === "workspace" && s.name === session.scenario)
    ? { icon: "🗺️", title: session.scenario }
    : town(session.scenario || session.label);
  const [stateText, stateClass] = STATE_PILL[status?.state ?? ""] ?? ["Connecting…", "bg-slate-100 text-slate-700"];
  const selection = snap?.selection && selected.kind !== "none" ? snap.selection : null;
  const recording = Boolean(status?.recording);

  return (
    <div className="flex min-h-screen flex-col md:h-screen">
      <header className="flex flex-wrap items-center gap-x-6 gap-y-2 bg-white px-4 py-3 shadow-sm">
        <Logo />
        <div className="flex items-center gap-2 text-lg font-semibold text-slate-800">
          <span aria-hidden>{t.icon}</span> {t.title}
          {isReplay && <span className="rounded-full bg-violet-100 px-2 py-0.5 text-xs font-medium text-violet-800">recording</span>}
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-3 text-sm">
          <span className={`rounded-full px-3 py-1 font-medium ${stateClass}`}>{stateText}</span>
          {recording && <span className="flex items-center gap-1 rounded-full bg-red-100 px-3 py-1 font-medium text-red-800"><span className="h-2 w-2 animate-pulse rounded-full bg-red-600" />Recording</span>}
          <span className="text-slate-700" title="Simulated time">⏰ <b className="text-lg">{clock(status?.time ?? 0)}</b></span>
          <span className="text-slate-700">🚗 <b className="text-lg">{fmt(status?.vehicles ?? 0, 0)}</b> cars</span>
          <button onClick={() => setHelp(true)} className={pill(false)}>❓ Help</button>
        </div>
      </header>

      <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 bg-slate-50 px-4 py-2">
        <button onClick={() => send({ type: playing ? "pause" : "play" })} disabled={!snap?.connected} title="Space"
          className={`rounded-full px-6 py-2.5 text-lg font-bold text-white shadow transition disabled:opacity-50 ${playing ? "bg-amber-500 hover:bg-amber-600" : "bg-emerald-600 hover:bg-emerald-500"}`}>
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>
        <div className="flex flex-wrap gap-1" role="group" aria-label="Speed">
          {SPEEDS.map((s) => (
            <button key={s.label} onClick={() => send({ type: "set_speed", steps_per_second: s.sps })}
              className={pill((status?.steps_per_second ?? null) === s.sps)} title={s.sps ? `${s.sps}× real time` : "As fast as the computer can"}>
              {s.icon} {s.label}
            </button>
          ))}
        </div>
        {signalised.length > 0 && !isReplay && (
          <label className="flex items-center gap-2 text-sm text-slate-700">
            <span>Signal control</span>
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
        <div className="ml-auto flex flex-wrap gap-2">
          {!isReplay && (
            <button onClick={() => send({ type: "record", on: !recording })} className={pill(recording, "bg-red-600 text-white ring-red-600")}>
              {recording ? "⏹ Stop recording" : "⏺ Record"}
            </button>
          )}
          <Downloads session={session} lastRecording={status?.last_recording ?? null} isReplay={isReplay} />
          {!isReplay && (
            <button onClick={() => confirm("Start this map again from the beginning?") && send({ type: "reset" })} className={pill(false)}>🔄 Restart</button>
          )}
          <button onClick={goHome} className={pill(false)}>🏠 Choose another map</button>
        </div>
      </div>
      {status?.last_recording && !recording && (
        <div className="flex items-center gap-3 bg-emerald-50 px-4 py-1.5 text-sm text-emerald-900">
          ✅ Recording saved as <b>{status.last_recording}</b>.
          <a className="underline" href={downloads.replay(status.last_recording)}>Download it</a>
          <span className="text-emerald-700">or watch it later from the start screen (Advanced options).</span>
        </div>
      )}
      {snap?.error && <div className="bg-red-50 px-4 py-1.5 text-sm text-red-800">⚠️ {snap.error}</div>}

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
              selectedRoad={selected.kind === "road" ? selected.id : null}
              route={snap?.selection?.id === selected.id ? remainingRoute(snap.selection) : null}
              onSelect={select}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-lg text-slate-600">Loading the map…</div>
          )}
          {hint && geometry && (
            <button onClick={() => setHint(false)} className="absolute left-3 top-3 max-w-xs rounded-xl bg-white/95 px-3 py-2 text-left text-sm text-slate-700 shadow">
              👆 <b>Click a car</b> to follow it, or <b>click a junction</b> to see its traffic light. Scroll to zoom, drag to move.
              <span className="ml-1 text-slate-400">✕</span>
            </button>
          )}
          {geometry && (
            <Legend busStops={(geometry.bus_stops?.id.length ?? 0) > 0} coloring={coloring} lanes={laneInfo ? LANE_VIEWS[laneInfo.code] ?? laneInfo.label : null} />
          )}
        </section>

        <aside className="flex w-full flex-col border-t border-slate-200 bg-slate-100 md:w-96 md:border-l md:border-t-0">
          <div className="flex gap-1 p-2" role="tablist">
            {([["traffic", "📊 Traffic"], ["compare", "🏁 Compare"], ["advanced", "⚙️ Advanced"]] as const).map(([k, label]) => (
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
                  <EventLog snap={snap} geometry={geometry} onPick={select} />
                </>
              ) : (
                <p className="text-slate-500">Connecting…</p>
              ))}
            {tab === "compare" && <RacePanel meta={meta} scenarios={scenarios} />}
            {tab === "advanced" && snap && (
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
                      <option value="veh:destination">Vehicles by where they are going</option>
                      <option value="veh:signals">Traffic lights (cars greyed out)</option>
                      {meta.lane_metrics.map((m) => (
                        <option key={m.code} value={`lane:${m.code}`}>Roads: {LANE_VIEWS[m.code] ?? m.label}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <TechnicalDetails snap={snap} selection={selection} />
                <p className="px-1 text-xs text-slate-500">
                  UrbanFlow v{meta.version} · workspace {meta.project} · session {session.id} · API docs at{" "}
                  <a className="underline" href="/api/v1/docs" target="_blank" rel="noreferrer">/api/v1/docs</a>
                </p>
              </>
            )}
          </div>
        </aside>
      </main>
      {helpPanel}
    </div>
  );
}
