// Colour scales and palettes for the map.

const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];

/** Viridis colour for t in [0, 1] (clamped); NaN gives a neutral no-data grey. */
export function viridis(t: number): string {
  if (!Number.isFinite(t)) return "rgba(100,116,139,0.35)";
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  const a = VIRIDIS[i]!;
  const b = VIRIDIS[i + 1]!;
  const c = (k: number) => Math.round(a[k]! + f * (b[k]! - a[k]!));
  return `rgb(${c(0)},${c(1)},${c(2)})`;
}

/** Red (stopped) -> yellow -> green (free flow) for a speed ratio in [0, 1]. */
export function speedColor(ratio: number): string {
  const r = Math.min(1, Math.max(0, ratio));
  return `hsl(${Math.round(r * 130)}, 80%, 55%)`;
}

/** Experiment variants, in spec order (validated categorical slots, fixed order; a
 * variant keeps its colour whatever is filtered or sorted). Past eight, charts fold. */
export const SERIES_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];

/** Vehicle colours by class (validated categorical slots; the legend names each). */
export const VCLASS_COLOR: Record<string, string> = {
  car: "#2a78d6",
  bus: "#eda100",
  truck: "#4a3aa7",
  emergency: "#e34948",
};

/** Where a vehicle is heading (by side of town; the legend names each). */
export const SIDE_COLOR: Record<string, string> = {
  north: "#2a78d6",
  east: "#eda100",
  south: "#4a3aa7",
  west: "#1baa7a",
};

/** Cars greyed out so the traffic lights stand out. */
export const DIM_VEHICLE = "#cbd5e1";

/** The map's light theme. */
export const MAP = {
  grass: "#cfe8c4",
  road: "#5f6670",
  junction: "#6b7280",
  marking: "rgba(255,255,255,0.9)",
  stopLine: "#ffffff",
  outline: "#ffffff",
  lightRing: "#1f2937",
  route: "rgba(250,204,21,0.55)",
  routeEnd: "#eab308",
  select: "#facc15",
};

// signal state codes: r 0, y 1, g 2, G 3, 255 unsignalised
export const SIGNAL_COLOR: Record<number, string> = {
  0: "#ef4444",
  1: "#facc15",
  2: "#4ade80",
  3: "#16a34a",
};
