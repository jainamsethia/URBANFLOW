// Plain-language names, icons and helpers for the kid-friendly workbench.

import type { Geometry, Point } from "./api";

export interface Town {
  icon: string;
  title: string;
  blurb: string;
}

const TOWNS: Record<string, Town> = {
  single_intersection: { icon: "🚦", title: "One crossing", blurb: "Four roads meet at one traffic light." },
  grid_3x3: { icon: "🏘️", title: "Small town", blurb: "Nine crossings in a neat grid." },
  grid_4x4: { icon: "🏙️", title: "Big town", blurb: "Sixteen crossings and lots of cars." },
  corridor: { icon: "🛣️", title: "Main street", blurb: "Lights timed so cars catch green after green." },
  emergency: { icon: "🚑", title: "Ambulance rescue", blurb: "Lights turn green for ambulances and cars move over." },
  rush_hour: { icon: "⏰", title: "Rush hour", blurb: "Traffic builds up to a busy peak, then calms down." },
};

export function town(name: string): Town {
  return TOWNS[name] ?? { icon: "🗺️", title: name.replaceAll("_", " "), blurb: "A town saved in this workspace." };
}

export const TOWN_ORDER = Object.keys(TOWNS);

export interface Brain {
  icon: string;
  title: string;
  blurb: string;
}

const BRAINS: Record<string, Brain> = {
  fixed_time: { icon: "⏱️", title: "Timer", blurb: "Lights change on a clock, no matter what." },
  actuated: { icon: "👀", title: "Sensor", blurb: "Stays green while cars keep coming." },
  max_pressure: { icon: "🧠", title: "Smart", blurb: "Gives green to the busiest roads." },
  webster: { icon: "📐", title: "Planner", blurb: "Works out the best timing in advance." },
  preemption: { icon: "🚑", title: "Ambulance helper", blurb: "A timer that turns green for ambulances." },
  external: { icon: "🎮", title: "Remote control", blurb: "A person or a program decides." },
};

export function brain(name: string): Brain {
  return BRAINS[name] ?? { icon: "🤖", title: name.replaceAll("_", " "), blurb: "A custom controller." };
}

/** Controllers a child can pick (external is for programs, not people). */
export function brainChoices(available: string[]): string[] {
  return Object.keys(BRAINS).filter((b) => b !== "external" && available.includes(b));
}

export const VEHICLE: Record<string, { icon: string; name: string }> = {
  car: { icon: "🚗", name: "Car" },
  bus: { icon: "🚌", name: "Bus" },
  truck: { icon: "🚚", name: "Truck" },
  emergency: { icon: "🚑", name: "Ambulance" },
};

export const SPEEDS: { label: string; icon: string; sps: number | null }[] = [
  { icon: "🐢", label: "Slow", sps: 2 },
  { icon: "🚶", label: "Normal", sps: 10 },
  { icon: "🐇", label: "Fast", sps: 50 },
  { icon: "🚀", label: "Super fast", sps: null },
];

export interface Mood {
  face: string;
  text: string;
  color: string;
}

/** How traffic feels, from the share of cars standing still (status palette + face + words). */
export function mood(active: number | null | undefined, halting: number | null | undefined): Mood {
  const n = active ?? 0;
  if (n <= 0) return { face: "🛣️", text: "The roads are empty", color: "#52514e" };
  const share = (halting ?? 0) / n;
  if (share < 0.2) return { face: "😀", text: "Traffic is flowing nicely", color: "#0ca30c" };
  if (share < 0.4) return { face: "🙂", text: "A little busy", color: "#fab219" };
  if (share < 0.6) return { face: "😟", text: "Lots of cars waiting", color: "#ec835a" };
  return { face: "😫", text: "Traffic jam!", color: "#d03b3b" };
}

export function kmh(ms: unknown): string {
  return typeof ms === "number" && Number.isFinite(ms) ? `${Math.round(ms * 3.6)} km/h` : "–";
}

export function seconds(s: unknown): string {
  if (typeof s !== "number" || !Number.isFinite(s)) return "–";
  const t = Math.round(s);
  return t < 60 ? `${t} s` : `${Math.floor(t / 60)} min ${t % 60} s`;
}

export function clock(t: number): string {
  const s = Math.max(0, Math.floor(t));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return `${h ? `${h}:` : ""}${String(m).padStart(h ? 2 : 1, "0")}:${String(sec).padStart(2, "0")}`;
}

// ---------------------------------------------------------------------------- compass
const COMPASS = [
  { name: "east", arrow: "⬅️" },
  { name: "north", arrow: "⬇️" },
  { name: "west", arrow: "➡️" },
  { name: "south", arrow: "⬆️" },
];

function centroid(pts: Point[]): Point {
  let x = 0, y = 0;
  for (const p of pts) { x += p[0]; y += p[1]; }
  return [x / Math.max(1, pts.length), y / Math.max(1, pts.length)];
}

function compass(dx: number, dy: number): (typeof COMPASS)[number] {
  const a = Math.atan2(dy, dx); // 0 = east, counter-clockwise
  const k = ((Math.round(a / (Math.PI / 2)) % 4) + 4) % 4;
  return COMPASS[k]!;
}

/** Which side of a junction a road comes from ("north", with the arrow cars drive). */
export function approachSide(g: Geometry, junction: string, road: string): { name: string; arrow: string } {
  const j = g.intersections.id.indexOf(junction);
  const r = g.roads.id.indexOf(road);
  if (j < 0 || r < 0) return { name: road, arrow: "•" };
  const [cx, cy] = centroid(g.roads.surfaces[r]!);
  const [jx, jy] = g.intersections.point[j]!;
  return compass(cx - jx, cy - jy);
}

/** Which edge of town a road leads to ("the east side"). */
export function townSide(g: Geometry, road: unknown): string {
  const r = typeof road === "string" ? g.roads.id.indexOf(road) : -1;
  if (r < 0) return "somewhere in town";
  const [mx, my] = centroid(g.intersections.point);
  const [cx, cy] = centroid(g.roads.surfaces[r]!);
  return `the ${compass(cx - mx, cy - my).name} side of town`;
}

/** The incoming side of every movement of a junction. */
export function movementSides(g: Geometry, junction: string): Map<string, { name: string; arrow: string }> {
  const out = new Map<string, { name: string; arrow: string }>();
  g.movements.id.forEach((id, i) => {
    if (g.movements.intersection[i] === junction) out.set(id, approachSide(g, junction, g.movements.from_road[i]!));
  });
  return out;
}
