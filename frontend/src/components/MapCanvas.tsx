// The live map: a Canvas2D renderer with a cached static layer, lane heat maps, signal heads
// and per-vehicle interpolation between frames. Nothing per-frame goes through React state.

import { useEffect, useRef } from "react";
import type { Geometry, LaneMetricInfo, Point } from "../lib/api";
import { DIM_VEHICLE, MAP, SIDE_COLOR, SIGNAL_COLOR, VCLASS_COLOR, speedColor, viridis } from "../lib/colors";
import { sideOfTown } from "../lib/friendly";
import type { SessionController } from "../lib/session";

/** How vehicles are painted: by kind, by speed, by where they are heading, or greyed out
 * so the traffic lights (drawn large) stand out. */
export type VehicleColoring = "type" | "speed" | "destination" | "signals";
export type Pick = "vehicle" | "intersection" | "road" | "none";

interface Props {
  geometry: Geometry;
  controller: SessionController;
  coloring: VehicleColoring;
  laneMetric: LaneMetricInfo | null;
  selectedVehicle: string | null;
  selectedIntersection: string | null;
  selectedRoad: string | null;
  /** Roads still ahead of the selected vehicle (highlighted; the last is its destination). */
  route: string[] | null;
  onSelect: (kind: Pick, id: string) => void;
}

const MINI_W = 180, MINI_H = 120;

/** Distance from (x, y) to the polyline ``pts``. */
function distToPath(pts: Point[], x: number, y: number): number {
  let best = Infinity;
  for (let i = 0; i + 1 < pts.length; i++) {
    const [ax, ay] = pts[i]!, [bx, by] = pts[i + 1]!;
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const t = len2 > 0 ? Math.min(1, Math.max(0, ((x - ax) * dx + (y - ay) * dy) / len2)) : 0;
    best = Math.min(best, Math.hypot(ax + t * dx - x, ay + t * dy - y));
  }
  return best;
}

interface View {
  cx: number;
  cy: number;
  scale: number; // px per metre
}

function bounds(g: Geometry): [number, number, number, number] {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  const add = (p: Point) => {
    x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]);
    x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]);
  };
  g.roads.surfaces.forEach((s) => s.forEach(add));
  g.intersections.point.forEach(add);
  return [x0, y0, x1, y1];
}

export function MapCanvas(props: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const miniRef = useRef<HTMLCanvasElement>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const viewRef = useRef<View | null>(null);
  const staticRef = useRef<{ canvas: HTMLCanvasElement; key: string } | null>(null);
  const fitRef = useRef<() => void>(() => {});

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const g = props.geometry;
    const [bx0, by0, bx1, by1] = bounds(g);
    const maxSpeed = 16;
    const vtypes = g.vehicle_types;
    const signalMov = g.signal_heads.movement;
    const signalPos = g.signal_heads.position;
    let raf = 0;
    let prevIndex: Map<number, number> = new Map();
    let prevFrameSeq = -1;
    // where each vehicle was last drawn (interpolated), so clicks pick what is on screen
    let drawnUid = new Uint32Array(0), drawnX = new Float64Array(0), drawnY = new Float64Array(0);
    const roadIndex = new Map(g.roads.id.map((id, i) => [id, i]));
    const destSide = g.roads.id.map((_, r) => sideOfTown(g, r)); // by destination road index
    const mini = miniRef.current!;
    const miniCtx = mini.getContext("2d")!;
    const miniScale = Math.min(MINI_W / Math.max(1, bx1 - bx0), MINI_H / Math.max(1, by1 - by0));
    const miniAt = (x: number, y: number): [number, number] => [
      MINI_W / 2 + (x - (bx0 + bx1) / 2) * miniScale,
      MINI_H / 2 - (y - (by0 + by1) / 2) * miniScale,
    ];
    let miniStatic: HTMLCanvasElement | null = null;

    const fit = () => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const scale = 0.92 * Math.min(w / Math.max(1, bx1 - bx0), h / Math.max(1, by1 - by0));
      viewRef.current = { cx: (bx0 + bx1) / 2, cy: (by0 + by1) / 2, scale };
    };
    fitRef.current = fit;
    fit();

    const toScreen = (v: View, w: number, h: number) => (x: number, y: number): [number, number] => [
      (x - v.cx) * v.scale + w / 2,
      h / 2 - (y - v.cy) * v.scale,
    ];

    const path = (c: CanvasRenderingContext2D, pts: Point[], T: (x: number, y: number) => [number, number], close: boolean) => {
      c.beginPath();
      pts.forEach((p, i) => {
        const [sx, sy] = T(p[0], p[1]);
        if (i === 0) c.moveTo(sx, sy);
        else c.lineTo(sx, sy);
      });
      if (close) c.closePath();
    };

    const drawStatic = (w: number, h: number, dpr: number, v: View) => {
      const key = `${w}x${h}@${dpr}:${v.cx.toFixed(2)},${v.cy.toFixed(2)},${v.scale.toFixed(5)}`;
      if (staticRef.current?.key === key) return staticRef.current.canvas;
      const off = staticRef.current?.canvas ?? document.createElement("canvas");
      off.width = w * dpr;
      off.height = h * dpr;
      const c = off.getContext("2d")!;
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.fillStyle = MAP.grass;
      c.fillRect(0, 0, w, h);
      const T = toScreen(v, w, h);
      c.fillStyle = MAP.road;
      for (const s of g.roads.surfaces) { path(c, s, T, true); c.fill(); }
      c.fillStyle = MAP.junction;
      g.intersections.polygons.forEach((poly, i) => {
        if (g.intersections.kind[i] === "boundary" || poly.length < 3) return;
        path(c, poly, T, true);
        c.fill();
      });
      c.lineWidth = Math.max(0.5, 0.15 * v.scale);
      g.lane_markings.path.forEach((p, i) => {
        c.strokeStyle = g.lane_markings.dashed[i] ? "rgba(255,255,255,0.7)" : MAP.marking;
        c.setLineDash(g.lane_markings.dashed[i] ? [3 * v.scale, 6 * v.scale] : []);
        path(c, p, T, false);
        c.stroke();
      });
      c.setLineDash([]);
      c.strokeStyle = MAP.stopLine;
      c.lineWidth = Math.max(1.5, 0.5 * v.scale);
      for (const p of g.stop_lines.path) { path(c, p, T, false); c.stroke(); }
      // bus stops: a sign at the curb
      const stops = g.bus_stops?.position ?? [];
      if (stops.length) {
        const size = Math.max(12, 2.6 * v.scale);
        c.font = `${size}px sans-serif`;
        c.textAlign = "center";
        c.textBaseline = "middle";
        for (const [x, y] of stops) {
          const [sx, sy] = T(x, y);
          c.fillText("🚏", sx, sy);
        }
      }
      staticRef.current = { canvas: off, key };
      return off;
    };

    // overview of the whole map with the visible area, when the map does not fit the view
    const drawMinimap = (w: number, h: number, v: View) => {
      const fits = (bx1 - bx0) * v.scale <= w * 1.05 && (by1 - by0) * v.scale <= h * 1.05;
      mini.style.display = fits ? "none" : "block";
      if (fits) return;
      const dpr = window.devicePixelRatio || 1;
      if (!miniStatic) {
        miniStatic = document.createElement("canvas");
        miniStatic.width = MINI_W * dpr;
        miniStatic.height = MINI_H * dpr;
        const c = miniStatic.getContext("2d")!;
        c.setTransform(dpr, 0, 0, dpr, 0, 0);
        c.fillStyle = "rgba(255,255,255,0.92)";
        c.fillRect(0, 0, MINI_W, MINI_H);
        c.fillStyle = MAP.road;
        for (const s of g.roads.surfaces) {
          c.beginPath();
          s.forEach((pt, i) => (i ? c.lineTo(...miniAt(pt[0], pt[1])) : c.moveTo(...miniAt(pt[0], pt[1]))));
          c.closePath();
          c.fill();
          c.lineWidth = 1;
          c.strokeStyle = MAP.road;
          c.stroke();
        }
      }
      if (mini.width !== MINI_W * dpr) {
        mini.width = MINI_W * dpr;
        mini.height = MINI_H * dpr;
      }
      miniCtx.setTransform(1, 0, 0, 1, 0, 0);
      miniCtx.drawImage(miniStatic, 0, 0);
      miniCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const [x0, y0] = miniAt(v.cx - w / 2 / v.scale, v.cy + h / 2 / v.scale);
      const [x1, y1] = miniAt(v.cx + w / 2 / v.scale, v.cy - h / 2 / v.scale);
      miniCtx.strokeStyle = MAP.routeEnd;
      miniCtx.lineWidth = 2;
      miniCtx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    };

    const render = () => {
      raf = requestAnimationFrame(render);
      const p = propsRef.current;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (w === 0 || h === 0) return; // collapsed layout: nothing to draw yet
      if (!(viewRef.current!.scale > 0)) fit(); // first fitted while the canvas had no size
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      const v = viewRef.current!;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(drawStatic(w, h, dpr, v), 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const T = toScreen(v, w, h);
      const cur = p.controller.curr;
      const prv = p.controller.prev;
      if (!cur) return;
      const f = cur.frame;

      // lane heat map
      if (p.laneMetric && f.laneValues) {
        const lv = f.laneValues;
        let hi = p.laneMetric.domain?.[1] ?? 0;
        if (!p.laneMetric.domain) for (let i = 0; i < lv.length; i++) if (Number.isFinite(lv[i]!)) hi = Math.max(hi, lv[i]!);
        const lo = p.laneMetric.domain?.[0] ?? 0;
        const span = Math.max(1e-9, hi - lo);
        ctx.lineCap = "butt";
        g.links.path.forEach((pts, i) => {
          if (g.links.kind[i] !== "lane") return;
          const val = lv[i]!;
          ctx.strokeStyle = viridis((val - lo) / span);
          ctx.globalAlpha = Number.isFinite(val) ? 0.85 : 0.25;
          ctx.lineWidth = Math.max(1, g.links.lane_width[i]! * 0.8 * v.scale);
          path(ctx, pts, T, false);
          ctx.stroke();
        });
        ctx.globalAlpha = 1;
      }

      // selected vehicle's remaining route
      if (p.route?.length) {
        ctx.lineWidth = 2;
        p.route.forEach((id, k) => {
          const s = g.roads.surfaces[roadIndex.get(id) ?? -1];
          if (!s) return;
          const last = k === p.route!.length - 1;
          path(ctx, s, T, true);
          ctx.fillStyle = MAP.route;
          ctx.fill();
          if (last) { ctx.strokeStyle = MAP.routeEnd; ctx.lineWidth = 3; ctx.stroke(); }
        });
      }

      // selected road
      if (p.selectedRoad) {
        const s = g.roads.surfaces[roadIndex.get(p.selectedRoad) ?? -1];
        if (s) {
          path(ctx, s, T, true);
          ctx.fillStyle = MAP.route;
          ctx.fill();
          ctx.strokeStyle = MAP.routeEnd;
          ctx.lineWidth = 3;
          ctx.stroke();
        }
      }

      const drawSignals = (r: number) => {
        if (!f.signals) return;
        for (let i = 0; i < signalMov.length; i++) {
          const state = f.signals[signalMov[i]!]!;
          const color = SIGNAL_COLOR[state];
          if (!color) continue;
          const [sx, sy] = T(signalPos[i]![0], signalPos[i]![1]);
          ctx.beginPath();
          ctx.arc(sx, sy, r, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.lineWidth = r > 6 ? 2 : 1;
          ctx.strokeStyle = r > 6 ? MAP.lightRing : MAP.outline;
          ctx.stroke();
        }
      };
      const signalsMode = p.coloring === "signals";
      if (!signalsMode) drawSignals(Math.max(3.5, 1.5 * v.scale));

      // interpolation set-up: index of every uid in the previous frame
      if (prv && prv.frame.seq !== prevFrameSeq) {
        prevIndex = new Map();
        const pu = prv.frame.uid;
        for (let i = 0; i < pu.length; i++) prevIndex.set(pu[i]!, i);
        prevFrameSeq = prv.frame.seq;
      }
      const now = performance.now();
      let alpha = 1;
      const stepGap = prv ? f.step - prv.frame.step : 0;
      if (prv && stepGap > 0 && stepGap <= 3) {
        const dtFrames = Math.min(1000, Math.max(16, cur.at - prv.at));
        alpha = Math.min(1, Math.max(0, (now - cur.at) / dtFrames));
      }

      // vehicles
      const vehicles = p.controller.vehicles;
      if (drawnUid.length !== f.n) {
        drawnUid = new Uint32Array(f.n);
        drawnX = new Float64Array(f.n);
        drawnY = new Float64Array(f.n);
      }
      for (let i = 0; i < f.n; i++) {
        const uid = f.uid[i]!;
        const meta = vehicles.get(uid);
        let x = f.xy[2 * i]!, y = f.xy[2 * i + 1]!, hd = f.heading[i]!;
        if (alpha < 1 && prv) {
          const j = prevIndex.get(uid);
          if (j !== undefined) {
            const px = prv.frame.xy[2 * j]!, py = prv.frame.xy[2 * j + 1]!;
            if (Math.hypot(x - px, y - py) < 60) {
              x = px + alpha * (x - px);
              y = py + alpha * (y - py);
              let d = hd - prv.frame.heading[j]!;
              d = ((d + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
              hd = prv.frame.heading[j]! + alpha * d;
            }
          }
        }
        const vt = meta ? vtypes[meta.type] : undefined;
        const len = vt?.length ?? 5, wid = vt?.width ?? 1.8;
        drawnUid[i] = uid;
        drawnX[i] = x;
        drawnY[i] = y;
        const [sx, sy] = T(x, y);
        ctx.save();
        ctx.translate(sx, sy);
        ctx.rotate(-hd);
        const vclass = vt?.vclass ?? "car";
        const blink = vclass === "emergency" && Math.floor(now / 350) % 2 === 0;
        const paint = (): string =>
          p.coloring === "speed"
            ? speedColor(f.speed[i]! / maxSpeed)
            : p.coloring === "destination"
              ? SIDE_COLOR[destSide[meta?.dest ?? -1] ?? ""] ?? DIM_VEHICLE
              : signalsMode
                ? DIM_VEHICLE
                : blink ? "#2563eb" : VCLASS_COLOR[vclass] ?? vt?.color ?? VCLASS_COLOR.car!;
        ctx.fillStyle = paint();
        const pxLen = Math.max(9, len * v.scale), pxWid = Math.max(5, wid * v.scale);
        const selectedCar = meta !== undefined && meta.id === p.selectedVehicle;
        if (selectedCar) {
          ctx.beginPath();
          ctx.arc(-pxLen / 2, 0, pxLen * 0.9 + 6, 0, 2 * Math.PI);
          ctx.fillStyle = "rgba(250,204,21,0.45)";
          ctx.fill();
          ctx.lineWidth = 2;
          ctx.strokeStyle = MAP.select;
          ctx.stroke();
          ctx.fillStyle = paint();
        }
        ctx.beginPath();
        ctx.roundRect(-pxLen, -pxWid / 2, pxLen, pxWid, Math.min(pxWid / 2, 3));
        ctx.fill();
        ctx.lineWidth = 0.75;
        ctx.strokeStyle = MAP.outline;
        ctx.stroke();
        ctx.restore();
      }

      if (signalsMode) drawSignals(Math.max(6, 3.2 * v.scale));

      drawMinimap(w, h, v);

      // selected intersection ring
      if (p.selectedIntersection) {
        const k = g.intersections.id.indexOf(p.selectedIntersection);
        if (k >= 0) {
          const [sx, sy] = T(g.intersections.point[k]![0], g.intersections.point[k]![1]);
          ctx.beginPath();
          ctx.arc(sx, sy, Math.max(10, (g.intersections.radius[k]! + 4) * v.scale), 0, 2 * Math.PI);
          ctx.strokeStyle = "#1f2937";
          ctx.lineWidth = 3;
          ctx.setLineDash([8, 5]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    };
    raf = requestAnimationFrame(render);

    // ------------------------------------------------------------------ interaction
    let drag: { x: number; y: number; cx: number; cy: number; moved: boolean } | null = null;
    const onDown = (e: MouseEvent) => {
      const v = viewRef.current!;
      drag = { x: e.clientX, y: e.clientY, cx: v.cx, cy: v.cy, moved: false };
    };
    const onMove = (e: MouseEvent) => {
      if (!drag) return;
      const v = viewRef.current!;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      viewRef.current = { ...v, cx: drag.cx - dx / v.scale, cy: drag.cy + dy / v.scale };
    };
    const onUp = (e: MouseEvent) => {
      if (!drag) return; // the press did not start on the map
      const wasDrag = drag.moved;
      drag = null;
      if (wasDrag) return;
      const rect = canvas.getBoundingClientRect();
      const v = viewRef.current!;
      const wx = (e.clientX - rect.left - rect.width / 2) / v.scale + v.cx;
      const wy = (rect.height / 2 - (e.clientY - rect.top)) / v.scale + v.cy;
      const p = propsRef.current;
      // a click inside a junction's box (or on its lights when zoomed out) picks the
      // traffic light; anywhere else the nearest drawn vehicle
      let k = -1, kd = Infinity;
      g.intersections.point.forEach((pt, i) => {
        if (g.intersections.kind[i] === "boundary") return;
        const d = Math.hypot(pt[0] - wx, pt[1] - wy);
        if (d < Math.max(g.intersections.radius[i]!, 10 / v.scale) && d < kd) { kd = d; k = i; }
      });
      if (k >= 0) { p.onSelect("intersection", g.intersections.id[k]!); return; }
      let best = -1, bestD = Math.max(4, 18 / v.scale);
      for (let i = 0; i < drawnUid.length; i++) {
        const d = Math.hypot(drawnX[i]! - wx, drawnY[i]! - wy);
        if (d < bestD) { bestD = d; best = i; }
      }
      const meta = best >= 0 ? p.controller.vehicles.get(drawnUid[best]!) : undefined;
      if (meta) {
        p.onSelect("vehicle", meta.id);
        return;
      }
      // else the road under the click (its lanes), if any
      let road = -1, roadD = Math.max(2, 8 / v.scale);
      g.links.path.forEach((pts, i) => {
        if (g.links.kind[i] !== "lane") return;
        const d = distToPath(pts, wx, wy) - g.links.lane_width[i]! / 2;
        if (d < roadD) { roadD = d; road = i; }
      });
      if (road >= 0) p.onSelect("road", g.links.owner[road]!);
      else p.onSelect("none", "");
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const v = viewRef.current!;
      const mx = e.clientX - rect.left - rect.width / 2;
      const my = rect.height / 2 - (e.clientY - rect.top);
      const wx = mx / v.scale + v.cx, wy = my / v.scale + v.cy;
      const scale = Math.min(200, Math.max(0.02, v.scale * Math.exp(-e.deltaY * 0.0015)));
      viewRef.current = { scale, cx: wx - mx / scale, cy: wy - my / scale };
    };
    const onMini = (e: MouseEvent) => {
      const rect = mini.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const v = viewRef.current!;
      viewRef.current = {
        ...v,
        cx: (bx0 + bx1) / 2 + (mx - MINI_W / 2) / miniScale,
        cy: (by0 + by1) / 2 - (my - MINI_H / 2) / miniScale,
      };
    };
    mini.addEventListener("click", onMini);
    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("dblclick", fit);
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) return;
      if (e.key === "f" || e.key === "F") fit();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      cancelAnimationFrame(raf);
      canvas.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("dblclick", fit);
      window.removeEventListener("keydown", onKey);
      mini.removeEventListener("click", onMini);
      staticRef.current = null;
    };
  }, [props.geometry, props.controller]);

  return (
    <div className="relative h-full w-full">
      <canvas
        ref={canvasRef}
        className="h-full w-full cursor-crosshair"
        role="application"
        aria-label="Simulation map: drag to pan, scroll to zoom, click a vehicle, junction or road"
      />
      <canvas
        ref={miniRef}
        style={{ width: MINI_W, height: MINI_H, display: "none" }}
        className="absolute bottom-3 right-3 cursor-pointer rounded-lg shadow ring-1 ring-slate-300"
        aria-label="Overview of the whole map: click to move there"
        title="The whole map: click to move there"
      />
      <button
        onClick={() => fitRef.current()}
        className="absolute right-3 top-3 rounded-full bg-white/95 px-3 py-1.5 text-sm font-medium text-slate-800 shadow hover:bg-white"
        title="Show the whole map (F, or double-click the map)"
      >
        🔍 Show the whole map
      </button>
    </div>
  );
}
