"""AT-13: right of way at the three intersection kinds (plan V, F.3, public API only).

Two deterministic vehicles on crossing paths of a 1-lane 4-arm junction (arms 200 m, dt
0.5, ``debug_checks=True``, so the zone-exclusivity invariant I5 and every other debug
invariant is checked at every step). Together, the vehicle with the right of way reaches
its exit road at its time alone on the network and the other one yields: it gets there
more than 1 s later than alone. Arrival order compares the times each would enter its
connector alone.

* signalised, permissive (g) left turn E -> S against the opposing protected (G) straight
  W -> E: the left turner reaches the line first but yields;
* priority junction (E-W major): the minor N -> S vehicle arrives first but yields to the
  major W -> E one;
* uncontrolled junction: first come, first served, whichever of the two arrives first.
"""

from __future__ import annotations

import pytest

from urbanflow import ScenarioBuilder, Simulation

pytestmark = pytest.mark.acceptance

ARMS = {"E": (1, 0), "N": (0, 1), "W": (-1, 0), "S": (0, -1)}
DET = {"speed_factor": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}}
DT = 0.5
Trip = tuple[str, float, list[str]]  # (id, depart time, route)


def _times(kind: str, trips: list[Trip]) -> dict[str, tuple[float, float]]:
    """Per trip: when it first shows up on a connector and on its exit road."""
    b = ScenarioBuilder("right_of_way", dt=DT, duration=120.0)
    b.vehicle_type("det", **DET)
    b.intersection("J", (0.0, 0.0), kind=kind)
    for name, (dx, dy) in ARMS.items():
        b.boundary(name, (dx * 200.0, dy * 200.0))
        b.road(f"{name}_in", name, "J")
        b.road(f"{name}_out", "J", name)
    for vid, depart, route in trips:
        b.trip(vid, depart, route=route, vehicle_type="det")
    signalised = kind == "signalized"
    sim = Simulation(
        b.build(), controllers={"J": "external"} if signalised else None, debug_checks=True
    )
    if signalised:  # W -> E protected, the opposing left E -> S permissive
        ew = next(p for p in sim.signals.phases("J") if p.green.get("W_in->E_out") == "G")
        assert ew.green["E_in->S_out"] == "g"
        sim.signals.set_phase("J", ew.index)
    enter: dict[str, float] = {}
    out: dict[str, float] = {}
    while len(out) < len(trips):
        sim.step()
        for vid, _, route in trips:
            car = sim.vehicles.get(vid)
            if car is not None and car.connector is not None:
                enter.setdefault(vid, sim.time)
            if vid not in out and car is not None and car.road == route[-1]:
                out[vid] = sim.time
        assert not sim.done, f"not every vehicle got through: {out}"
    assert sim.metrics.summary()["vehicles.teleported"] == 0
    return {vid: (enter[vid], out[vid]) for vid in out}


def _right_of_way(kind: str, first: Trip, second: Trip) -> bool:
    """``first`` goes (not delayed) and ``second`` yields; returns whether ``second`` would
    have reached the junction first."""
    alone = {t[0]: _times(kind, [t])[t[0]] for t in (first, second)}
    both = _times(kind, [first, second])
    assert both[first[0]][1] == pytest.approx(alone[first[0]][1], abs=DT)  # not delayed
    assert both[second[0]][1] > alone[second[0]][1] + 1.0  # yields
    return alone[second[0]][0] < alone[first[0]][0]


LEFT: Trip = ("left", 0.0, ["E_in", "S_out"])
STRAIGHT: Trip = ("straight", 6.0, ["W_in", "E_out"])


def test_at13_permissive_left_yields_to_the_opposing_protected_straight() -> None:
    assert _right_of_way("signalized", STRAIGHT, LEFT)  # g yields though first


def test_at13_minor_yields_to_major_at_a_priority_junction() -> None:
    minor: Trip = ("minor", 0.0, ["N_in", "S_out"])
    major: Trip = ("major", 1.0, ["W_in", "E_out"])
    assert _right_of_way("priority", major, minor)  # the minor yields though first


@pytest.mark.parametrize(("t_we", "t_ns"), [(0.0, 1.0), (1.0, 0.0)])
def test_at13_first_come_first_served_at_an_uncontrolled_junction(t_we: float, t_ns: float) -> None:
    we: Trip = ("we", t_we, ["W_in", "E_out"])
    ns: Trip = ("ns", t_ns, ["N_in", "S_out"])
    first, second = (we, ns) if t_we < t_ns else (ns, we)
    assert not _right_of_way("uncontrolled", first, second)  # the earlier one goes
