from __future__ import annotations

import numpy as np
import pytest

from urbanflow import Simulation, generate
from urbanflow.core.errors import ScenarioValidationError


def _rates(scenario: object, flow: str) -> list[float]:
    flows = scenario.resolved.demand.flows  # type: ignore[attr-defined]
    return [f.rate for f in flows if f.id.rsplit(":", 1)[0] == flow]


def test_slices_follow_the_peak_and_direction() -> None:
    s = generate("rush_hour")  # corridor base: 900 veh/h at each main-street end
    flows = s.resolved.demand.flows
    assert len(flows) == 36 * 24 and flows[0].id == "W_J0:far:0"
    assert (flows[0].begin, flows[0].end) == (0.0, 300.0)
    east = _rates(s, "W_J0:straight")  # bearing 0 = the peak direction: x1.5
    west = _rates(s, "E_J4:straight")  # against it: x0.5
    base = 900 * 0.8  # straight share of the corridor's turn ratios
    m = [0.3 + 0.7 * max(0.0, 1 - abs(300 * k + 150 - 3600) / 1800) for k in range(24)]
    assert east == pytest.approx([base * 1.5 * x for x in m])
    assert west == pytest.approx([base * 0.5 * x for x in m])
    flat = generate("rush_hour", peak_direction_deg=None)
    assert _rates(flat, "W_J0:straight") == _rates(flat, "E_J4:straight")


@pytest.mark.parametrize(
    "params", [{"base": "rush_hour"}, {"slice": 9000}, {"off_peak_factor": 1.5}]
)
def test_bad_params(params: dict[str, object]) -> None:
    with pytest.raises(ScenarioValidationError):
        generate("rush_hour", **params)


def test_more_departures_at_the_peak() -> None:
    s = generate("rush_hour", duration=1800, peak_time=900, peak_width=450, slice=150, base="grid")
    trips = Simulation(s, seed=0).run().tables["trips"]
    dep = trips["depart_time"]
    peak = np.count_nonzero((dep >= 675) & (dep < 1125))
    early = np.count_nonzero(dep < 450)
    assert peak > 2 * early


def test_webster_design_demand_is_time_averaged() -> None:
    # unit profile: the slices carry the base rates, so the run-averaged design demand
    # (what Webster times for) must equal the unsliced scenario's
    flat = generate("rush_hour", off_peak_factor=1.0, peak_direction_deg=None, duration=1800)
    base = generate("corridor", duration=1800)
    q_flat = Simulation(flat)._engine.design_flows()
    q_base = Simulation(base)._engine.design_flows()
    assert q_flat == pytest.approx(q_base)
