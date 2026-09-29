# Road network

The simulator never reads the scenario JSON while it runs. The network compiler
(`urbanflow.network.compile_network`) turns the *resolved* scenario (every derived field
filled in, see [Scenario format](scenario-format.md)) into a `CompiledNetwork`: immutable,
integer-indexed NumPy arrays of geometry, topology and conflicts that the engine reads every
step. This page covers the conventions, the compilation rules, the conflict model and the
render geometry that clients draw.

```python
import urbanflow
from urbanflow.network import compile_network
from urbanflow.visualization import render_geometry

scenario = urbanflow.Scenario.load(urbanflow.bundled("single_intersection"))
net = compile_network(scenario)
print(net.n_lanes, net.n_conn, net.n_conflicts)  # 16 16 52
xy, heading = net.point_at([0, 3], [10.0, 50.0])  # positions along two links
geometry = render_geometry(net).model_dump(mode="json")
```

## Conventions

| Topic | Rule |
|---|---|
| Units | Metres, seconds, m/s. Headings are radians, counter-clockwise from +x. |
| Coordinates | Scenario files use world coordinates (x east, y north). Compiled arrays, frames and render geometry use **local** coordinates: world = local + `origin`, where `origin` is the minimum of the network's bounding box. This keeps float32 frame coordinates precise far from (0, 0). |
| Lanes | A road's `points` are its **median edge**. Lane 0 is the median (innermost) lane, lane `n-1` the curb lane. |
| Driving side | `drive_side="right"` stacks lanes to the right of travel, `"left"` to the left; the compiler mirrors lane offsets (a sign flip) and derivation mirrors lane mapping and turn sides. |
| Ids | Lanes `"{road}_{k}"`, connectors `"{lane_in}->{lane_out}"`, movements `"{from_road}->{to_road}"`. |
| Links | Lanes and connectors share one index space: lanes first (lane `k` of road `r` is link `road_lane_start[r] + k`), then connectors in movement order. |
| Bearings | Approaches are ordered clockwise from north (N, E, S, W on a 4-way junction), then by lane index. |

## Compilation rules

1. **Resolve.** The input is `Scenario.resolved`; the compiler applies no defaults.
2. **Trim roads.** At each road end touching a non-boundary intersection *j*, the reference
   polyline is cut at arc length τ = max(0, r<sub>j</sub> − ‖p<sub>end</sub> − c<sub>j</sub>‖).
   The cut follows the polyline around bends (CityFlow cuts along the last segment and
   overshoots). Lanes shorter than 5 m raise **E802**.
3. **Lane polylines.** Lane *k* is offset o<sub>k</sub> = Σ<sub>i&lt;k</sub> w<sub>i</sub> + w<sub>k</sub>/2
   from the median edge toward the curb. Interior vertices use the miter offset
   o/cos(φ/2); above a miter factor of 4 the vertex is bevelled into two points. A lane
   that folds back on itself (its offset exceeds the radius of a bend) raises **E905**.
4. **Connectors.** Each connection joins the end of its incoming lane to the start of its
   outgoing lane. An explicit `shape` is used as-is with its end points snapped to the
   lanes; otherwise a cubic Bézier approximates a circular arc (handles
   h = 4/3 · tan(θ/4) · R, sampled every π/16 of turning). The connector speed limit is
   min(v<sub>in</sub>, v<sub>out</sub>, √(a<sub>lat</sub> R<sub>min</sub>)) with
   a<sub>lat</sub> = 2 m/s² and R<sub>min</sub> the smallest three-point circumradius:
   about 4.5 m/s for a left turn of radius 10 m. A connector of zero length (lanes meeting
   at one point, e.g. an intersection with radius 0) raises **E905**.
5. **Conflicts** between the connectors of each intersection (next section).
6. **Static ranks** (`mov_static_rank`): signalised movements 0 (their rank comes from the
   signal state: G = 3, g = 2); at priority intersections major straight and near-side
   turns 3, major far-side turns and U-turns 2, minor movements 1; uncontrolled 1.
7. **Render geometry** (last section).
8. **Road graph.** `urbanflow.network.graph.build_road_graph(net)` returns a
   `networkx.DiGraph` with a node per road and an edge per movement, weighted by the length
   and free-flow time of the target road, built in sorted order for deterministic ties.

Compiling the same scenario twice gives identical arrays, and every array is read-only, so
one `CompiledNetwork` is safely shared across resets and simulations.

## Positions along links

`net.point_at(links, pos)` returns local points and headings for many vehicles with one
`searchsorted`. All link polylines are concatenated in `pts` with a globally monotone arc
key `s_global = link_base[l] + cumulative arc`, where consecutive links are separated by a
1 m gap (`link_base[l+1] = link_base[l] + link_length[l] + 1`). The position is clamped to
`[0, link_length]` and located in its link's segment.

## Conflicts

For every pair of connectors (a, b) at one intersection whose bounding boxes overlap:

| Kind | When | Zone on each connector |
|---|---|---|
| diverging (2) | same `from_lane` | [0, d<sub>sep</sub>] |
| merging (1) | same `to_lane` | [L − d<sub>sep</sub>, L] |
| crossing (0) | the polylines cross | [s − h, s + h] around the first crossing |

d<sub>sep</sub> is the first arc length at which the other connector is farther than
W<sub>max</sub> = 3.0 m (the widest vehicle, 2.6 m, plus a 0.4 m margin), capped at the
connector length. For crossings, h = (W/2)/sin θ′ + (W/2)/tan θ′ with θ′ = min(θ, π − θ),
the **acute** angle between the centre lines: a permissive left turn crossing the
opposing straight at about 133° gets a 6.9 m zone, not the 1.3 m the obtuse angle would
give. Below 15° the closed form blows up, so the zone is the interval around the crossing
where the other connector stays within W<sub>max</sub>. Zones are clipped to the connector.

A pair that crosses twice (only possible with explicit shapes) gets one zone covering both
crossings and a **W304** warning. Diverging conflicts are compiled for geometry and
reporting only: vehicles leaving one lane stay in order and car-following separates them.

On the bundled `single_intersection` (4 arms, 2 lanes, lane mapping L{0} S{0,1} R{1}) this
gives 36 crossing, 8 merging and 8 diverging conflicts. Each connector's conflicts are
listed in `conn_conf` sorted by where its zone starts on that connector.

## Diagnostics

`compile_network` raises `ScenarioValidationError` for errors and keeps warnings in
`net.report.warnings`. `urbanflow.check`, which `urbanflow validate` runs by default
(`--no-deep` skips it) and `Simulation` runs before it starts, compiles every valid file,
so these appear together with the file-level checks.

| Code | Path | Meaning |
|---|---|---|
| E802 | `network.roads[i]` | Trimmed lanes would be shorter than 5 m (also checked by validation). |
| E905 | `network.roads[i]` or `network.intersections[j]...` | A lane reverses direction, or a connector has zero length. |
| E806 | `network.roads[i].lanes[k]` | A lane is shorter than `length + min_gap` of a vehicle type routed onto it (first road of a demand source, or a connector target along its route; OD sources use their shortest path). |
| W302 | `network.intersections[j].signal.phases[p]` | Two movements that cross are both protected (G) in one phase. |
| W304 | `network.intersections[j]` | Two connectors cross twice; their zones were merged. |

## Render geometry

`urbanflow.visualization.render_geometry(net)` returns a `RenderGeometry` (a pydantic
model with a JSON Schema). It is what the REST API serves and replays store as
`geometry.json`. Coordinates are local and rounded to 1 mm; large collections are
**columnar** (one list per field) so clients can build binary layer attributes directly.

| Field | Contents |
|---|---|
| `version`, `geometry_crc` | Format version (1) and the crc32 of the scenario content hash; frames carry the same crc. |
| `origin`, `bbox` | World position of the local origin and the local bounding box. |
| `links` | Every link in link-index order: `id`, `kind` (lane/connector), `owner` (road of a lane, movement of a connector), `lane_width`, `path`. |
| `movements` | Movement-index order: `id`, `intersection`, `from_road`, `to_road`, `turn`, connector `links`. |
| `signal_heads` | One per movement at the stop line of its median-most lane (heads sharing a lane are spread across it); frames colour them by `signals[movement]`. |
| `roads` | `id` and the road `surfaces` (the union of the lane quads). |
| `intersections` | `id`, `kind`, `point`, `radius` and `polygons`: the convex hull of the lane-end corners and a 16-point radius disc (empty for boundary nodes). |
| `lane_markings` | `road`, `dashed` and `path`: dashed separators between lanes, solid median and curb edges. |
| `stop_lines` | A segment across the end of every lane entering a non-boundary intersection. |
| `vehicle_types` | `id`, `vclass`, `length`, `width`, `color`; vehicle messages index this list. |
