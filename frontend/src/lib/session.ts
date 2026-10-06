// SessionController: the WebSocket data plane of one session. Frames never go through
// React state; React components subscribe to a small, throttled snapshot instead.

import { decodeFrame, type FrameView } from "./ufb";
import { wsUrl } from "./api";

export interface VehicleMeta {
  id: string;
  type: number;
  dest: number;
}

export interface Status {
  state: string;
  step: number;
  time: number;
  end_time: number | null;
  steps_per_second: number | null;
  vehicles: number;
  lane_metric: number;
  controllers: Record<string, string>;
  error: string;
}

export interface Selection {
  kind: "vehicle" | "intersection";
  id: string;
  available: boolean;
  detail?: Record<string, unknown>;
}

export type MetricSeries = Record<string, number[]>;

export interface Snapshot {
  connected: boolean;
  status: Status | null;
  metrics: Record<string, number | null>;
  series: MetricSeries;
  selection: Selection | null;
  error: string | null;
}

const SERIES = ["time", "active", "speed_mean", "queue_mean", "throughput_vph", "halting", "space_mean_speed"];
const MAX_POINTS = 720;

export class SessionController {
  readonly vehicles = new Map<number, VehicleMeta>();
  prev: { frame: FrameView; at: number } | null = null;
  curr: { frame: FrameView; at: number } | null = null;
  snapshot: Snapshot = { connected: false, status: null, metrics: {}, series: {}, selection: null, error: null };
  private ws: WebSocket | null = null;
  private listeners = new Set<() => void>();
  private closed = false;
  private retry = 0;

  constructor(readonly sessionId: string) {
    for (const k of SERIES) this.snapshot.series[k] = [];
    this.connect();
  }

  seedHistory(history: Record<string, (number | null)[]>): void {
    const series: MetricSeries = {};
    for (const k of SERIES) series[k] = (history[k] ?? []).map((v) => (v == null ? NaN : v)).slice(-MAX_POINTS);
    this.update({ series });
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  send(msg: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  dispose(): void {
    this.closed = true;
    this.ws?.close();
    this.listeners.clear();
  }

  private update(patch: Partial<Snapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    for (const fn of this.listeners) fn();
  }

  private connect(): void {
    const ws = new WebSocket(wsUrl(this.sessionId), ["urbanflow.v1"]);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => {
      this.retry = 0;
      this.update({ connected: true, error: null });
    };
    ws.onclose = (ev) => {
      this.update({ connected: false });
      if (this.closed) return;
      if (ev.code === 4404 || ev.code === 4410) {
        this.update({ error: "Session ended" });
        return;
      }
      const delay = Math.min(8000, 500 * 2 ** this.retry++);
      setTimeout(() => !this.closed && this.connect(), delay);
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") this.onText(JSON.parse(ev.data));
      else this.onFrame(decodeFrame(ev.data as ArrayBuffer));
    };
  }

  private onFrame(frame: FrameView): void {
    // drop frames that reference unknown vehicles (registry deltas always come first)
    this.prev = this.curr;
    this.curr = { frame, at: performance.now() };
  }

  private onText(m: Record<string, any>): void {
    switch (m.type) {
      case "vehicles_added": {
        if (m.reset) {
          this.vehicles.clear();
          this.prev = this.curr = null;
        }
        const uids: number[] = m.uids;
        uids.forEach((u, i) => this.vehicles.set(u, { id: m.ids[i], type: m.types[i], dest: m.dests[i] }));
        break;
      }
      case "vehicles_removed":
        for (const u of m.uids as number[]) this.vehicles.delete(u);
        break;
      case "status":
        this.update({ status: m as Status });
        break;
      case "metrics": {
        const values = m.values as Record<string, number | null>;
        const series: MetricSeries = {};
        const lastT = this.snapshot.series.time?.at(-1);
        const fresh = values.time != null && values.time !== lastT;
        for (const k of SERIES) {
          const arr = this.snapshot.series[k] ?? [];
          if (!fresh) {
            series[k] = arr;
            continue;
          }
          const v = values[k];
          // a reset rewinds time: start the series again
          const base = values.time != null && lastT != null && values.time < lastT ? [] : arr;
          series[k] = [...base, v == null ? NaN : v].slice(-MAX_POINTS);
        }
        this.update({ metrics: values, series });
        break;
      }
      case "selection":
        this.update({ selection: m as Selection });
        break;
      case "error":
        this.update({ error: m.message });
        break;
      case "hello":
        break;
    }
  }
}
