// The map editor: draw junctions and roads, set lights and traffic, fix problems, save and
// try the map. Every change is an operation the server applies and validates (so the map
// shown is always exactly what will run); undo/redo keep the documents client-side.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type Meta, type ImportIssue } from "../lib/api";
import { brain, brainChoices } from "../lib/friendly";
import { pill } from "./StartScreen";

type Pt = [number, number];
interface Ix {
  id: string;
  point: Pt;
  kind?: string;
  signal?: { controller?: { type?: string }; yellow?: number; phases?: { duration?: number }[] };
}
interface Road {
  id: string;
  from: string;
  to: string;
  lanes: unknown[];
  speed_limit?: number;
  points?: Pt[];
}
interface Flow {
  id: string;
  origin?: string;
  destination?: string;
  route?: string[];
  rate?: number;
}
export interface Draft {
  format: string;
  version: string;
  meta: { name: string; description?: string };
  simulation?: { duration?: number; [k: string]: unknown };
  network: { intersections: Ix[]; roads: Road[]; [k: string]: unknown };
  demand?: { flows?: Flow[]; [k: string]: unknown };
  [k: string]: unknown;
}
type Op = Record<string, unknown>;
type Tool = "select" | "junction" | "road" | "erase";
type Sel = { kind: "junction" | "road"; id: string } | null;

export const BLANK_MAP: Draft = {
  format: "urbanflow.scenario",
  version: "1.0",
  meta: { name: "my-map" },
  network: { intersections: [], roads: [] },
};

const LANE_WIDTH = 3.2;
const SNAP = 5;
const NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const KINDS: [string, string][] = [
  ["auto", "✨ Automatic"],
  ["signalized", "🚦 Traffic light"],
  ["priority", "⚠️ Give way (main road first)"],
  ["uncontrolled", "↔️ No control (first come, first served)"],
  ["boundary", "🚪 Edge of the map (cars enter and leave)"],
];
const KIND_COLOR: Record<string, string> = {
  signalized: "#1f2937",
  priority: "#eda100",
  uncontrolled: "#64748b",
  boundary: "#ffffff",
};
const SPEED_KMH = [20, 30, 40, 50, 60, 80, 100];

/** Each junction's type: as set, or as UrbanFlow derives it (plan E.7 §1.5 step 2). */
function junctionKinds(d: Draft): Map<string, string> {
  const ins = new Map<string, number>(), outs = new Map<string, number>(), nb = new Map<string, Set<string>>();
  const add = (m: Map<string, number>, k: string) => m.set(k, (m.get(k) ?? 0) + 1);
  for (const r of d.network.roads) {
    add(outs, r.from);
    add(ins, r.to);
    if (!nb.has(r.from)) nb.set(r.from, new Set());
    if (!nb.has(r.to)) nb.set(r.to, new Set());
    nb.get(r.from)!.add(r.to);
    nb.get(r.to)!.add(r.from);
  }
  const out = new Map<string, string>();
  for (const ix of d.network.intersections) {
    const n = nb.get(ix.id)?.size ?? 0;
    const derived = n <= 1 ? "boundary" : (ins.get(ix.id) ?? 0) <= 2 && (outs.get(ix.id) ?? 0) <= 2 && n === 2 ? "uncontrolled" : "signalized";
    out.set(ix.id, ix.kind ?? derived);
  }
  return out;
}

function freeId(d: Draft, stem: string, taken: string[] = []): string {
  const used = new Set([...d.network.intersections.map((i) => i.id), ...taken]);
  let n = 1;
  while (used.has(`${stem}${n}`)) n++;
  return `${stem}${n}`;
}

/** Plain words for the problems of a map that is still being drawn. */
function advice(i: ImportIssue): string {
  if (i.path === "network.intersections" && i.code === "E004") return "Add junctions: pick 🛣️ Draw road and click on the map.";
  if (i.path === "network.roads" && i.code === "E004") return "Connect junctions with roads: with 🛣️ Draw road, click one junction after another.";
  return i.message;
}

/** Which entity an issue's path names, if any. */
function issueTarget(d: Draft, path: string): Sel {
  const m = /^network\.(intersections|roads)\[(\d+)\]/.exec(path);
  if (!m) return null;
  const list = m[1] === "intersections" ? d.network.intersections : d.network.roads;
  const item = list[Number(m[2])];
  return item ? { kind: m[1] === "intersections" ? "junction" : "road", id: item.id } : null;
}

// ------------------------------------------------------------------ the drawing board
function Board({
  draft,
  kinds,
  selected,
  tool,
  roadStart,
  onClick,
  onMoveJunction,
}: {
  draft: Draft;
  kinds: Map<string, string>;
  selected: Sel;
  tool: Tool;
  roadStart: string | null;
  onClick: (hit: Sel, x: number, y: number) => void;
  onMoveJunction: (id: string, x: number, y: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const view = useRef({ cx: 0, cy: 0, scale: 0 });
  const mouse = useRef<Pt | null>(null);
  const drag = useRef<{ x: number; y: number; cx: number; cy: number; junction: string | null; moved: boolean } | null>(null);
  const live = useRef({ draft, kinds, selected, tool, roadStart, onClick, onMoveJunction });
  live.current = { draft, kinds, selected, tool, roadStart, onClick, onMoveJunction };
  const dragPoint = useRef<Pt | null>(null);

  const fit = useCallback(() => {
    const c = ref.current;
    if (!c) return;
    const pts = live.current.draft.network.intersections.map((i) => i.point);
    const w = c.clientWidth || 800, h = c.clientHeight || 600;
    if (!pts.length) {
      view.current = { cx: 0, cy: 0, scale: 1 };
      return;
    }
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const scale = Math.min(4, 0.8 * Math.min(w / Math.max(50, x1 - x0), h / Math.max(50, y1 - y0)));
    view.current = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, scale };
  }, []);

  const toScreen = (x: number, y: number): Pt => {
    const c = ref.current!, v = view.current;
    return [(x - v.cx) * v.scale + c.clientWidth / 2, c.clientHeight / 2 - (y - v.cy) * v.scale];
  };
  const toWorld = (sx: number, sy: number): Pt => {
    const c = ref.current!, v = view.current;
    return [(sx - c.clientWidth / 2) / v.scale + v.cx, (c.clientHeight / 2 - sy) / v.scale + v.cy];
  };

  /** Screen polyline of a road, offset to its side of the street. */
  const roadLine = (r: Road, pts: Map<string, Pt>): Pt[] => {
    const a = dragPoint.current && drag.current?.junction === r.from ? dragPoint.current : pts.get(r.from);
    const b = dragPoint.current && drag.current?.junction === r.to ? dragPoint.current : pts.get(r.to);
    if (!a || !b) return [];
    const line = r.points && !drag.current?.junction ? r.points : [a, b];
    const half = (Math.max(1, r.lanes.length) * LANE_WIDTH) / 2;
    const out: Pt[] = [];
    for (let i = 0; i < line.length; i++) {
      const p = line[i]!, q = line[Math.min(i + 1, line.length - 1)]!, o = line[Math.max(i - 1, 0)]!;
      const [dx, dy] = i + 1 < line.length ? [q[0] - p[0], q[1] - p[1]] : [p[0] - o[0], p[1] - o[1]];
      const len = Math.hypot(dx, dy) || 1;
      out.push(toScreen(p[0] + (dy / len) * half, p[1] - (dx / len) * half));
    }
    return out;
  };

  const draw = useCallback(() => {
    const c = ref.current;
    if (!c) return;
    const { draft: d, kinds: k, selected: sel, roadStart: start } = live.current;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (!(view.current.scale > 0)) fit();
    if (c.width !== w * dpr || c.height !== h * dpr) {
      c.width = w * dpr;
      c.height = h * dpr;
    }
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#e7f0e1";
    ctx.fillRect(0, 0, w, h);
    // grid every 50 m
    const v = view.current;
    const step = 50 * v.scale >= 12 ? 50 : 250;
    ctx.strokeStyle = "rgba(15,23,42,0.06)";
    ctx.lineWidth = 1;
    const [wx0, wy0] = toWorld(0, h), [wx1, wy1] = toWorld(w, 0);
    for (let x = Math.floor(wx0 / step) * step; x <= wx1; x += step) {
      const [sx] = toScreen(x, 0);
      ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, h); ctx.stroke();
    }
    for (let y = Math.floor(wy0 / step) * step; y <= wy1; y += step) {
      const [, sy] = toScreen(0, y);
      ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(w, sy); ctx.stroke();
    }
    const pts = new Map(d.network.intersections.map((i) => [i.id, i.point] as [string, Pt]));
    // roads
    ctx.lineCap = "round";
    for (const r of d.network.roads) {
      const line = roadLine(r, pts);
      if (line.length < 2) continue;
      const width = Math.max(3, r.lanes.length * LANE_WIDTH * v.scale);
      const isSel = sel?.kind === "road" && sel.id === r.id;
      const stroke = (style: string, lw: number) => {
        ctx.strokeStyle = style;
        ctx.lineWidth = lw;
        ctx.beginPath();
        line.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.stroke();
      };
      if (isSel) stroke("#facc15", width + 6);
      stroke("#5f6670", width);
      // direction arrow at the middle
      const a = line[Math.floor((line.length - 1) / 2)]!, b = line[Math.floor((line.length - 1) / 2) + 1]!;
      const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
      const s = Math.max(4, Math.min(9, width * 0.6));
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.beginPath();
      ctx.moveTo(mx + Math.cos(ang) * s, my + Math.sin(ang) * s);
      ctx.lineTo(mx + Math.cos(ang + 2.5) * s, my + Math.sin(ang + 2.5) * s);
      ctx.lineTo(mx + Math.cos(ang - 2.5) * s, my + Math.sin(ang - 2.5) * s);
      ctx.fill();
    }
    // road being drawn
    const startPt = start ? pts.get(start) : undefined;
    if (startPt && mouse.current) {
      const [ax, ay] = toScreen(startPt[0], startPt[1]);
      ctx.setLineDash([8, 6]);
      ctx.strokeStyle = "#0369a1";
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(mouse.current[0], mouse.current[1]); ctx.stroke();
      ctx.setLineDash([]);
    }
    // junctions
    ctx.font = "12px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const ix of d.network.intersections) {
      const p = dragPoint.current && drag.current?.junction === ix.id ? dragPoint.current : ix.point;
      const [sx, sy] = toScreen(p[0], p[1]);
      const kind = k.get(ix.id) ?? "boundary";
      const isSel = sel?.kind === "junction" && sel.id === ix.id;
      if (isSel || ix.id === start) {
        ctx.beginPath(); ctx.arc(sx, sy, 15, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(250,204,21,0.55)"; ctx.fill();
      }
      ctx.beginPath(); ctx.arc(sx, sy, 8, 0, 2 * Math.PI);
      ctx.fillStyle = KIND_COLOR[kind] ?? "#64748b"; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = kind === "boundary" ? "#1f2937" : "#ffffff"; ctx.stroke();
      if (kind === "signalized") {
        ctx.beginPath(); ctx.arc(sx, sy, 3, 0, 2 * Math.PI); ctx.fillStyle = "#4ade80"; ctx.fill();
      }
      ctx.fillStyle = "#0f172a";
      ctx.fillText(ix.id, sx + 12, sy - 10);
    }
    if (!d.network.intersections.length) {
      ctx.fillStyle = "#475569";
      ctx.font = "16px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Pick 🛣️ Draw road, then click to place junctions and connect them.", w / 2, h / 2);
    }
  }, [fit]);

  useEffect(() => {
    draw();
  });

  useEffect(() => {
    const c = ref.current!;
    const hit = (sx: number, sy: number): Sel => {
      const { draft: d } = live.current;
      for (const ix of d.network.intersections) {
        const [jx, jy] = toScreen(ix.point[0], ix.point[1]);
        if (Math.hypot(jx - sx, jy - sy) <= 12) return { kind: "junction", id: ix.id };
      }
      const pts = new Map(d.network.intersections.map((i) => [i.id, i.point] as [string, Pt]));
      let best: Sel = null, bestD = Infinity;
      for (const r of d.network.roads) {
        const line = roadLine(r, pts);
        const tol = Math.max(6, (r.lanes.length * LANE_WIDTH * view.current.scale) / 2 + 3);
        for (let i = 0; i + 1 < line.length; i++) {
          const [ax, ay] = line[i]!, [bx, by] = line[i + 1]!;
          const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy || 1;
          const t = Math.max(0, Math.min(1, ((sx - ax) * dx + (sy - ay) * dy) / l2));
          const dist = Math.hypot(ax + t * dx - sx, ay + t * dy - sy);
          if (dist <= tol && dist < bestD) { bestD = dist; best = { kind: "road", id: r.id }; }
        }
      }
      return best;
    };
    const local = (e: MouseEvent): Pt => {
      const r = c.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    const onDown = (e: MouseEvent) => {
      const [sx, sy] = local(e);
      const h = hit(sx, sy);
      const v = view.current;
      drag.current = {
        x: sx, y: sy, cx: v.cx, cy: v.cy, moved: false,
        junction: live.current.tool === "select" && h?.kind === "junction" ? h.id : null,
      };
    };
    const onMove = (e: MouseEvent) => {
      const [sx, sy] = local(e);
      mouse.current = [sx, sy];
      const g = drag.current;
      if (g) {
        if (Math.abs(sx - g.x) + Math.abs(sy - g.y) > 3) g.moved = true;
        if (g.junction && g.moved) {
          const [wx, wy] = toWorld(sx, sy);
          dragPoint.current = [Math.round(wx / SNAP) * SNAP, Math.round(wy / SNAP) * SNAP];
        } else if (g.moved) {
          const v = view.current;
          view.current = { ...v, cx: g.cx - (sx - g.x) / v.scale, cy: g.cy + (sy - g.y) / v.scale };
        }
      }
      draw();
    };
    const onUp = (e: MouseEvent) => {
      const g = drag.current;
      drag.current = null;
      if (!g) return;
      const [sx, sy] = local(e);
      if (g.junction && g.moved && dragPoint.current) {
        const [x, y] = dragPoint.current;
        dragPoint.current = null;
        live.current.onMoveJunction(g.junction, x, y);
        return;
      }
      dragPoint.current = null;
      if (g.moved) return;
      const [wx, wy] = toWorld(sx, sy);
      live.current.onClick(hit(sx, sy), Math.round(wx / SNAP) * SNAP, Math.round(wy / SNAP) * SNAP);
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const [sx, sy] = local(e);
      const [wx, wy] = toWorld(sx, sy);
      const v = view.current;
      const scale = Math.min(20, Math.max(0.05, v.scale * Math.exp(-e.deltaY * 0.0015)));
      view.current = { scale, cx: wx - (sx - c.clientWidth / 2) / scale, cy: wy - (c.clientHeight / 2 - sy) / scale };
      draw();
    };
    const onResize = () => draw();
    c.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    c.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("resize", onResize);
    return () => {
      c.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      c.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", onResize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="relative h-full w-full">
      <canvas ref={ref} className={`h-full w-full ${tool === "select" ? "cursor-default" : "cursor-crosshair"}`}
        role="application" aria-label="Map editor: click to add junctions and roads, drag to move, scroll to zoom" />
      <button onClick={() => { fit(); draw(); }} className="absolute right-3 top-3 rounded-full bg-white/95 px-3 py-1.5 text-sm font-medium text-slate-800 shadow">
        🔍 Show the whole map
      </button>
    </div>
  );
}

// ------------------------------------------------------------------ the editor
export function Editor({
  meta,
  initial,
  initialName,
  onExit,
  onRun,
}: {
  meta: Meta;
  initial: Draft;
  initialName: string;
  onExit: () => void;
  onRun: (name: string) => void;
}) {
  const [draft, setDraft] = useState<Draft>(initial);
  const [past, setPast] = useState<Draft[]>([]);
  const [future, setFuture] = useState<Draft[]>([]);
  const [issues, setIssues] = useState<ImportIssue[]>([]);
  const [valid, setValid] = useState(false);
  const [tool, setTool] = useState<Tool>(initial.network.intersections.length ? "select" : "road");
  const [lanes, setLanes] = useState(1);
  const [selected, setSelected] = useState<Sel>(null);
  const [roadStart, setRoadStart] = useState<string | null>(null);
  const [name, setName] = useState(initialName);
  const [message, setMessage] = useState<{ text: string; bad?: boolean } | null>(null);
  const [busy, setBusy] = useState(false);
  const [json, setJson] = useState<string | null>(null);
  const [flow, setFlow] = useState({ origin: "", destination: "", rate: 300 });
  const kinds = useMemo(() => junctionKinds(draft), [draft]);

  const check = useCallback(async (d: Draft) => {
    const res = await api.ops(d, []);
    setIssues(res.issues);
    setValid(res.valid);
  }, []);
  useEffect(() => {
    check(initial).catch((e) => setMessage({ text: (e as Error).message, bad: true }));
  }, [check, initial]);

  /** Apply operations; the result becomes the map (undoable). */
  const apply = async (ops: Op[], next?: Draft) => {
    setBusy(true);
    setMessage(null);
    try {
      const res = await api.ops(next ?? draft, ops);
      setPast((p) => [...p.slice(-99), draft]);
      setFuture([]);
      setDraft(res.scenario as Draft);
      setIssues(res.issues);
      setValid(res.valid);
      return res.created;
    } catch (e) {
      setMessage({ text: (e as Error).message, bad: true });
      return null;
    } finally {
      setBusy(false);
    }
  };
  const undo = () => {
    const prev = past.at(-1);
    if (!prev) return;
    setFuture((f) => [draft, ...f]);
    setPast((p) => p.slice(0, -1));
    setDraft(prev);
    check(prev).catch(() => {});
  };
  const redo = () => {
    const next = future[0];
    if (!next) return;
    setPast((p) => [...p, draft]);
    setFuture((f) => f.slice(1));
    setDraft(next);
    check(next).catch(() => {});
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
      if (e.key === "Escape") { setRoadStart(null); setSelected(null); }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); if (e.shiftKey) redo(); else undo(); }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); redo(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const onBoardClick = async (hit: Sel, x: number, y: number) => {
    if (busy) return;
    if (tool === "select") return setSelected(hit);
    if (tool === "erase") {
      if (hit) await apply([{ op: hit.kind === "junction" ? "delete_intersection" : "delete_road", id: hit.id }]);
      setSelected(null);
      return;
    }
    if (tool === "junction") {
      if (hit?.kind === "junction") return setSelected(hit);
      const id = freeId(draft, "J");
      if (await apply([{ op: "add_intersection", id, point: [x, y] }])) setSelected({ kind: "junction", id });
      return;
    }
    // road tool: click junctions (or empty ground for a new one); each click continues the road
    const ops: Op[] = [];
    let target = hit?.kind === "junction" ? hit.id : null;
    if (!target) {
      target = freeId(draft, "J");
      ops.push({ op: "add_intersection", id: target, point: [x, y] });
    }
    if (roadStart && roadStart !== target) ops.push({ op: "add_road", from: roadStart, to: target, lanes });
    if (ops.length && !(await apply(ops))) return;
    setRoadStart(target);
  };

  const save = async (): Promise<boolean> => {
    if (!NAME.test(name)) {
      setMessage({ text: "Use letters, digits, '.', '_' or '-' for the name (starting with a letter or digit).", bad: true });
      return false;
    }
    setBusy(true);
    try {
      const doc = { ...draft, meta: { ...draft.meta, name } };
      await api.importScenario({ name, format: "urbanflow", scenario: doc, overwrite: true });
      setMessage({ text: `Saved as "${name}" in your workspace.` });
      return true;
    } catch (e) {
      setMessage({ text: (e as Error).message, bad: true });
      return false;
    } finally {
      setBusy(false);
    }
  };

  const ix = selected?.kind === "junction" ? draft.network.intersections.find((i) => i.id === selected.id) : undefined;
  const road = selected?.kind === "road" ? draft.network.roads.find((r) => r.id === selected.id) : undefined;
  const reverse = road ? draft.network.roads.find((r) => r.from === road.to && r.to === road.from) : undefined;
  const entries = draft.network.roads.filter((r) => kinds.get(r.from) === "boundary");
  const exits = draft.network.roads.filter((r) => kinds.get(r.to) === "boundary");
  const flows = draft.demand?.flows ?? [];
  const errors = issues.filter((i) => i.severity === "error");
  const warnings = issues.filter((i) => i.severity !== "error");
  const card = "rounded-2xl bg-white p-4 shadow-sm ring-1 ring-slate-200";
  const field = "mt-1 w-full rounded-lg border border-slate-300 bg-white p-1.5 text-sm";
  const controller = ix?.signal?.controller?.type ?? "fixed_time";
  const green = ix?.signal?.phases?.[0]?.duration;
  const duration = Number(draft.simulation?.duration ?? 3600);
  const TOOLS: [Tool, string, string][] = [
    ["select", "👆 Select / move", "Click to select; drag a junction to move it"],
    ["road", "🛣️ Draw road", "Click junctions (or empty ground) one after the other; Esc to stop"],
    ["junction", "📍 Add junction", "Click to place a junction"],
    ["erase", "🧽 Erase", "Click a junction or road to remove it"],
  ];

  return (
    <div className="flex min-h-screen flex-col md:h-screen">
      <header className="flex flex-wrap items-center gap-2 bg-white px-4 py-3 shadow-sm">
        <span className="text-lg font-semibold text-slate-800">✏️ Map editor</span>
        <div className="flex flex-wrap gap-1" role="group" aria-label="Tool">
          {TOOLS.map(([t, label, tip]) => (
            <button key={t} title={tip} aria-pressed={tool === t} onClick={() => { setTool(t); setRoadStart(null); }} className={pill(tool === t)}>{label}</button>
          ))}
        </div>
        {tool === "road" && (
          <label className="flex items-center gap-1 text-sm text-slate-700">
            lanes each way
            <select className="rounded-full border border-slate-300 px-2 py-1" value={lanes} onChange={(e) => setLanes(+e.target.value)}>
              {[1, 2, 3, 4].map((n) => <option key={n}>{n}</option>)}
            </select>
          </label>
        )}
        <div className="ml-auto flex flex-wrap gap-1">
          <button onClick={undo} disabled={!past.length} title="Ctrl+Z" className={`${pill(false)} disabled:opacity-40`}>↶ Undo</button>
          <button onClick={redo} disabled={!future.length} title="Ctrl+Y" className={`${pill(false)} disabled:opacity-40`}>↷ Redo</button>
          <button onClick={() => setJson(json === null ? JSON.stringify(draft, null, 2) : null)} className={pill(json !== null)}>{"{ }"} JSON</button>
          <button onClick={onExit} className={pill(false)}>🏠 Back</button>
        </div>
      </header>
      {message && <div className={`px-4 py-1.5 text-sm ${message.bad ? "bg-red-50 text-red-800" : "bg-emerald-50 text-emerald-900"}`} style={{ whiteSpace: "pre-wrap" }}>{message.text}</div>}

      <main className="flex min-h-0 flex-1 flex-col md:flex-row">
        <section className="relative h-[60vh] min-h-80 min-w-0 md:h-auto md:flex-1">
          {json === null ? (
            <Board draft={draft} kinds={kinds} selected={selected} tool={tool} roadStart={roadStart}
              onClick={onBoardClick}
              onMoveJunction={(id, x, y) => apply([{ op: "move_intersection", id, point: [x, y] }])} />
          ) : (
            <div className="flex h-full flex-col gap-2 p-3">
              <textarea aria-label="Scenario JSON" spellCheck={false} value={json} onChange={(e) => setJson(e.target.value)}
                className="min-h-0 flex-1 rounded-lg border border-slate-300 p-2 font-mono text-xs" />
              <div className="flex gap-2">
                <button className={pill(false)} onClick={async () => {
                  let parsed: Draft;
                  try { parsed = JSON.parse(json) as Draft; } catch (e) { setMessage({ text: `Not valid JSON: ${(e as Error).message}`, bad: true }); return; }
                  if (await apply([], parsed)) setJson(null);
                }}>✔ Use this JSON</button>
                <button className={pill(false)} onClick={() => setJson(null)}>Cancel</button>
              </div>
            </div>
          )}
        </section>

        <aside className="w-full space-y-3 overflow-y-auto border-t border-slate-200 bg-slate-100 p-3 md:w-96 md:border-l md:border-t-0">
          <div className={card}>
            <label className="block text-sm font-semibold text-slate-900">
              Map name
              <input className={field} value={name} onChange={(e) => setName(e.target.value)} />
            </label>
            <label className="mt-2 block text-sm text-slate-700">
              Run it for
              <select className={field} value={duration} onChange={(e) => apply([{ op: "set_simulation", duration: +e.target.value }])}>
                {[600, 1800, 3600, 7200].map((s) => <option key={s} value={s}>{s / 60} minutes</option>)}
                {![600, 1800, 3600, 7200].includes(duration) && <option value={duration}>{Math.round(duration / 60)} minutes</option>}
              </select>
            </label>
            <div className="mt-3 flex flex-wrap gap-2">
              <button disabled={busy || !valid} onClick={save} className={`${pill(false)} disabled:opacity-40`} title={valid ? "" : "Fix the problems first"}>💾 Save</button>
              <button disabled={busy || !valid} onClick={async () => (await save()) && onRun(name)}
                className="rounded-full bg-emerald-600 px-4 py-1.5 text-sm font-semibold text-white shadow disabled:opacity-40">▶ Try it</button>
            </div>
          </div>

          {ix && (
            <div className={card}>
              <div className="font-semibold text-slate-900">📍 Junction {ix.id}</div>
              <label className="mt-2 block text-sm text-slate-700">
                Type
                <select className={field} value={ix.kind ?? "auto"} onChange={(e) => apply([{ op: "update_intersection", id: ix.id, kind: e.target.value }])}>
                  {KINDS.map(([k, label]) => <option key={k} value={k}>{k === "auto" ? `${label} (${KINDS.find(([x]) => x === kinds.get(ix.id))?.[1] ?? ""})` : label}</option>)}
                </select>
              </label>
              {kinds.get(ix.id) === "signalized" && (
                <>
                  <label className="mt-2 block text-sm text-slate-700">
                    How the lights work
                    <select className={field} value={controller} onChange={(e) => apply([{ op: "set_controller", intersection: ix.id, controller: e.target.value }])}>
                      {[...new Set([controller, ...brainChoices(meta.controllers)])].map((c) => <option key={c} value={c}>{brain(c).icon} {brain(c).title}</option>)}
                    </select>
                  </label>
                  <label className="mt-2 block text-sm text-slate-700">
                    Green time per direction
                    <select className={field} value={green ?? 0} onChange={(e) => apply([{ op: "set_signal_timing", intersection: ix.id, green: +e.target.value }])}>
                      <option value={0} disabled>Automatic (30 s)</option>
                      {[10, 15, 20, 30, 45, 60].map((g) => <option key={g} value={g}>{g} s</option>)}
                    </select>
                  </label>
                </>
              )}
              <button className="mt-3 text-sm text-red-700 underline" onClick={() => apply([{ op: "delete_intersection", id: ix.id }]).then(() => setSelected(null))}>Remove this junction and its roads</button>
            </div>
          )}

          {road && (
            <div className={card}>
              <div className="font-semibold text-slate-900">🛣️ Road {road.id}</div>
              <div className="text-sm text-slate-600">from {road.from} to {road.to}</div>
              <label className="mt-2 block text-sm text-slate-700">
                Lanes
                <select className={field} value={road.lanes.length} onChange={(e) => apply([{ op: "update_road", id: road.id, lanes: +e.target.value }])}>
                  {[1, 2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </label>
              <label className="mt-2 block text-sm text-slate-700">
                Speed limit
                <select className={field} value={Math.round((road.speed_limit ?? 13.89) * 3.6)} onChange={(e) => apply([{ op: "update_road", id: road.id, speed_limit: +e.target.value / 3.6 }])}>
                  {[...new Set([...SPEED_KMH, Math.round((road.speed_limit ?? 13.89) * 3.6)])].sort((a, b) => a - b).map((k) => <option key={k} value={k}>{k} km/h</option>)}
                </select>
              </label>
              <div className="mt-3 flex flex-col items-start gap-1 text-sm">
                <button className="text-red-700 underline" onClick={() => apply([{ op: "delete_road", id: road.id }]).then(() => setSelected(null))}>Remove this direction</button>
                {reverse && <button className="text-red-700 underline" onClick={() => apply([{ op: "delete_road", id: road.id }, { op: "delete_road", id: reverse.id }]).then(() => setSelected(null))}>Remove both directions</button>}
              </div>
            </div>
          )}

          <div className={card}>
            <div className="font-semibold text-slate-900">🚗 Traffic</div>
            {flows.length === 0 && <p className="text-sm text-slate-600">No traffic yet. Cars come in and leave at the 🚪 edges of the map.</p>}
            <ul className="mt-1 space-y-1">
              {flows.map((f) => (
                <li key={f.id} className="flex items-center gap-2 text-sm">
                  <span className="min-w-0 flex-1 truncate" title={f.id}>{f.origin ?? f.route?.[0]} → {f.destination ?? f.route?.at(-1)}</span>
                  <input aria-label={`cars per hour of ${f.id}`} type="number" min={1} max={20000} defaultValue={f.rate ?? ""} key={`${f.id}:${f.rate}`}
                    className="w-20 rounded border border-slate-300 p-1 text-right"
                    onBlur={(e) => { const v = +e.target.value; if (v > 0 && v !== f.rate) apply([{ op: "update_flow", id: f.id, rate: v }]); }} />
                  <span className="text-xs text-slate-500">cars/h</span>
                  <button aria-label={`remove ${f.id}`} className="text-red-700" onClick={() => apply([{ op: "delete_flow", id: f.id }])}>✕</button>
                </li>
              ))}
            </ul>
            {entries.length > 0 && exits.length > 0 ? (
              <div className="mt-2 space-y-1 border-t border-slate-100 pt-2 text-sm">
                <div className="font-medium text-slate-800">Add traffic</div>
                <select aria-label="From" className={field} value={flow.origin} onChange={(e) => setFlow({ ...flow, origin: e.target.value })}>
                  <option value="">From…</option>
                  {entries.map((r) => <option key={r.id} value={r.id}>from {r.from} (road {r.id})</option>)}
                </select>
                <select aria-label="To" className={field} value={flow.destination} onChange={(e) => setFlow({ ...flow, destination: e.target.value })}>
                  <option value="">To…</option>
                  {exits.map((r) => <option key={r.id} value={r.id}>to {r.to} (road {r.id})</option>)}
                </select>
                <div className="flex items-center gap-2">
                  <input aria-label="Cars per hour" type="number" min={1} max={20000} value={flow.rate} onChange={(e) => setFlow({ ...flow, rate: +e.target.value })} className="w-24 rounded border border-slate-300 p-1 text-right" />
                  <span className="text-slate-600">cars per hour</span>
                  <button disabled={!flow.origin || !flow.destination || !(flow.rate > 0)} className={`${pill(false)} ml-auto disabled:opacity-40`}
                    onClick={() => apply([{ op: "add_flow", origin: flow.origin, destination: flow.destination, rate: flow.rate }])}>➕ Add</button>
                </div>
              </div>
            ) : (
              <p className="mt-1 text-xs text-slate-500">Draw roads to at least two edges of the map to add traffic.</p>
            )}
          </div>

          <div className={card}>
            <div className="font-semibold text-slate-900">{valid ? "✅ Ready to run" : `⚠️ ${errors.length} problem${errors.length === 1 ? "" : "s"} to fix`}</div>
            <ul className="mt-1 max-h-60 space-y-1 overflow-y-auto text-sm">
              {[...errors, ...warnings].map((i, k) => {
                const target = issueTarget(draft, i.path);
                return (
                  <li key={`${i.path}-${k}`}>
                    <button disabled={!target} onClick={() => target && setSelected(target)}
                      className={`w-full rounded-lg px-2 py-1 text-left ${i.severity === "error" ? "bg-red-50 text-red-900" : "bg-amber-50 text-amber-900"} ${target ? "hover:ring-1 hover:ring-slate-300" : ""}`}>
                      {i.severity === "error" ? "⛔" : "⚠️"} {advice(i)}
                      <span className="block text-xs opacity-70">{i.path}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        </aside>
      </main>
    </div>
  );
}
