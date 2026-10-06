"""AT-26: a saturated 4-way junction with every turn and exit spillback, 2 h (plan V, F.3).

Junction J (1 lane per arm, all turns 15/70/15 %) gets 400 veh/h per approach, above what
it serves. Its east exit ``E_out`` ends at a second junction J2 whose outflow ``E_slow``
has a 1 m/s speed limit, i.e. ~330 veh/h (IDM peak flow) against the 400 veh/h bound for
the east: the queue spills back over ``E_out`` into J, where don't-block-the-box and the
zone locks must keep every movement going. 7200 steps at dt 1 with ``debug_checks=True``
(zone exclusivity I5 and every other invariant, every step). Pass criteria: no invariant
violation, arrivals in every 5-minute window, no vehicle halting on a connector for more
than 60 s; ``forced_commits`` and the other safety counters are reported. About 25 s per
variant, two thirds of it the engine itself (~2 ms per step with ~110 vehicles running)
and a third the debug checks; shorter arms do not make it cheaper.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from urbanflow import ScenarioBuilder
from urbanflow.engine import Engine
from urbanflow.scenario import Scenario

pytestmark = [pytest.mark.acceptance, pytest.mark.slow]  # ~25 s per variant, in the default run

RATE = 400.0  # veh/h per approach
WINDOW = 300  # steps (5 min at dt 1)
STEPS = 7200


def _scenario(kind: str) -> Scenario:
    b = ScenarioBuilder("saturated", dt=1.0, duration=float(STEPS), seed=3)
    b.intersection("J", (0.0, 0.0), kind=kind)
    b.intersection("J2", (120.0, 0.0), kind="uncontrolled")
    for name, point in (("W", (-200.0, 0.0)), ("N", (0.0, 200.0)), ("S", (0.0, -200.0))):
        b.boundary(name, point)
        b.road(f"{name}_in", name, "J")
        b.road(f"{name}_out", "J", name)
    b.boundary("E", (420.0, 0.0))
    b.road("E_in", "J2", "J")
    b.road("E_out", "J", "J2")
    b.road("E_back", "E", "J2")
    b.road("E_slow", "J2", "E", speed_limit=1.0)  # the downstream bottleneck
    entry = {"W": ["W_in"], "N": ["N_in"], "S": ["S_in"], "E": ["E_back", "E_in"]}
    exit_ = {"W": ["W_out"], "N": ["N_out"], "S": ["S_out"], "E": ["E_out", "E_slow"]}
    straight = {"W": "E", "E": "W", "N": "S", "S": "N"}
    for a in entry:
        for z in exit_:
            if z != a:
                share = 0.7 if straight[a] == z else 0.15
                b.flow(f"{a}{z}", route=entry[a] + exit_[z], rate=RATE * share, arrival="poisson")
    return b.build()


@pytest.mark.parametrize("kind", ["uncontrolled", "signalized"])
def test_at26_saturated_junction_with_exit_spillback(
    make_engine: Callable[..., Engine], kind: str, record_property: Callable[[str, object], None]
) -> None:
    e = make_engine(_scenario(kind))  # debug_checks=True: an I5 overlap would raise
    net, veh = e.network, e.vehicles
    exit_lane = net.link_index["E_out_0"]
    windows: list[int] = []
    last = 0
    worst = 0.0  # longest continuous halt on a connector, s
    spill = 0.0  # longest halted queue on E_out, m
    for n in range(STEPS):
        e.step()
        run = veh.running()
        inside = run[veh.link[run] >= net.n_lanes]
        if inside.size:
            worst = max(worst, float(veh.stuck_time[inside].max()))
        queue = run[(veh.link[run] == exit_lane) & veh.halting[run]]
        if queue.size:
            rear = float((veh.pos[queue] - veh.length[queue]).min())
            spill = max(spill, float(net.link_length[exit_lane]) - rear)
        if (n + 1) % WINDOW == 0:
            windows.append(e.arrived - last)
            last = e.arrived
    counters = {
        "forced_commits": e.forced_commits,
        "red_runs": e.red_runs,
        "zone_conflicts": e.zone_conflicts,
        "safety_cap_violations": e.safety_cap_violations,
        "teleports": e.teleported,
        "arrived": e.arrived,
        "backlog": e.backlog,
    }
    for name, value in counters.items():
        record_property(name, value)
    print(f"AT-26 {kind}: {counters}, arrivals per 5 min {windows}")  # the report
    assert e.backlog > 0  # saturated
    assert spill > 0.9 * float(net.link_length[exit_lane])  # the exit queue reached J
    assert len(windows) == STEPS // WINDOW and min(windows) > 0
    assert worst <= 60.0


def _grid(kind: str, dt: float, seed: int) -> Scenario:
    """A 2x2 grid of 1-lane two-way roads (90 m) with a boundary on every outer arm and a
    40 veh/h flow (20 % trucks) between every pair of boundary roads."""
    length = 90.0
    b = ScenarioBuilder("grid_trucks", dt=dt, duration=900.0, seed=seed)
    for r in range(2):
        for c in range(2):
            b.intersection(f"J{r}{c}", (c * length, r * length), kind=kind)
    arms = {}
    for k in range(2):
        for name, point, junction in (
            (f"S{k}", (k * length, -length), f"J0{k}"),
            (f"N{k}", (k * length, 2 * length), f"J1{k}"),
            (f"W{k}", (-length, k * length), f"J{k}0"),
            (f"E{k}", (2 * length, k * length), f"J{k}1"),
        ):
            b.boundary(name, point)
            b.two_way(name, junction)
            arms[name] = junction
    for a, z in (("J00", "J01"), ("J10", "J11"), ("J00", "J10"), ("J01", "J11")):
        b.two_way(a, z)
    if kind == "signalized":
        b.signal_all()
    names = sorted(arms)
    for o in names:
        for d in names:
            if o != d:
                b.flow(
                    f"{o}{d}",
                    origin=f"{o}_{arms[o]}",
                    destination=f"{arms[d]}_{d}",
                    rate=40.0,
                    arrival="poisson",
                    type_mix={"car": 0.8, "truck": 0.2},
                )
    return b.build()


@pytest.mark.parametrize(("kind", "dt"), [("signalized", 1.0), ("signalized", 2.0)])
def test_grid_with_trucks_has_no_false_overlaps(
    make_engine: Callable[..., Engine], kind: str, dt: float
) -> None:
    """P4 review repro: a truck that just left a merging sibling hangs back ~10 m over it;
    a vehicle waiting at an earlier zone of its own connector "overlapped" it in the
    lookahead (invariant I4 raised; without debug checks the false negative gap fired the
    safety cap and inflated ``safety_cap_violations``). Every debug invariant runs at
    every step; about 3 s per variant."""
    e = make_engine(_grid(kind, dt, seed=0))  # debug_checks=True
    for _ in range(round(900.0 / dt)):
        e.step()
    assert e.arrived > 250
    assert e.safety_cap_violations == 0 and e.red_runs == 0
