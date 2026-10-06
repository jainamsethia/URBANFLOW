from __future__ import annotations

import pytest

from urbanflow import Simulation, generate
from urbanflow.core.errors import ScenarioValidationError


def test_layout_and_signal_timing() -> None:
    s = generate("corridor", intersections=3, cycle=90, main_share=0.6)
    summary = s.summary()
    assert summary["intersections"] == 3 + 2 + 6  # junctions, W/E, N/S per junction
    ixs = {ix.id: ix for ix in s.resolved.network.intersections}
    sig = ixs["J1"].signal
    assert sig is not None and [p.id for p in sig.phases] == ["main", "side"]
    assert [p.duration for p in sig.phases] == [round(82 * 0.6), round(82 - round(82 * 0.6))]
    j0 = ixs["J0"].signal
    assert j0 is not None
    main = dict(j0.phases[0].green)
    assert main["W_J0->J0_J1"] == "G" and main["W_J0->J0_N0"] == "g"  # straight / left
    assert "N0_J0->J0_S0" not in main and dict(j0.phases[1].green)["N0_J0->J0_S0"] == "G"
    offsets = [ixs[f"J{i}"].signal.controller.params.get("offset") for i in range(3)]  # type: ignore[union-attr]
    assert offsets == [0.0, round(250 / 16.67), round(500 / 16.67)]
    flat = generate("corridor", intersections=3, coordination="none")
    assert all(
        ix.signal.controller.params.get("offset") == 0.0
        for ix in flat.resolved.network.intersections
        if ix.signal is not None
    )


def test_green_wave_helps_through_traffic() -> None:
    def through_travel(coordination: str) -> float:
        sim = Simulation(generate("corridor", coordination=coordination, duration=1200), seed=3)
        trips = sim.run().tables["trips"]
        main = (trips["origin"] == "W_J0") & (trips["destination"] == "J4_E")
        return float(trips["travel_time"][main].mean())

    assert through_travel("green_wave") < 0.85 * through_travel("none")


@pytest.mark.parametrize("params", [{"cycle": 20}, {"spacing": 10}, {"main_share": 0.99}])
def test_bad_params(params: dict[str, object]) -> None:
    with pytest.raises(ScenarioValidationError):
        generate("corridor", **params)
