"""Snapshot -> step -> restore -> step on real runs (plan F.5, U.1 engine/state.py row)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from urbanflow import generate
from urbanflow.core.errors import SimulationError
from urbanflow.engine import Engine
from urbanflow.engine.state import load_state, save_state, state_digest

pytestmark = pytest.mark.integration

MakeEngine = Callable[..., Engine]
K = 150


def _run(e: Engine, n: int, commands: Callable[[Engine, int], None] | None = None) -> list[str]:
    out = []
    for i in range(n):
        if commands is not None:
            commands(e, i)
        e.step()
        out.append(e.digest())
    return out


def _commands(e: Engine, i: int) -> None:
    """The same interventions after the snapshot and after the restore."""
    if i == 10:
        e.commands.add_vehicle(route=["W_in", "E_out"], id="probe")
    if i == 30 and "probe" in e.vehicles.id_to_handle:
        e.commands.set_speed("probe", 4.0, duration=20.0)
    if i == 40 and "probe" in e.vehicles.id_to_handle:
        h = e.vehicles.handle_of("probe")
        on_w_in = e.network.link_ids[int(e.vehicles.link[h])].startswith("W_in_")
        if on_w_in and not e.vehicles.committed[h]:
            e.commands.set_route("probe", ["W_in", "N_out"])
    if i == 60 and e.signals.programs:
        e.commands.hold_phase("J", 1)
    if i == 110 and e.signals.programs:
        e.commands.release("J")


@pytest.mark.parametrize("kind", ["uncontrolled", "signalized"])
def test_snapshot_step_restore_step_gives_equal_digests(make_engine: MakeEngine, kind: str) -> None:
    e = make_engine(generate("single_intersection", kind=kind, demand_rate=800), seed=5)
    _run(e, 200)
    s = e.snapshot()
    locked = s.arrays["veh.lock_conn"] >= 0
    assert locked.any() or kind == "signalized"  # zone locks are part of the state
    before = e.digest()
    ahead = _run(e, K, _commands)
    assert e.teleported == 0
    e.restore(s)
    assert e.digest() == before
    assert _run(e, K, _commands) == ahead


def test_file_round_trip_continues_identically(make_engine: MakeEngine, tmp_path: Path) -> None:
    e = make_engine(generate("single_intersection", demand_rate=800), seed=2)
    _run(e, 120)
    e.commands.hold_phase("J", 1)  # a parked controller and a pending request
    path = save_state(tmp_path / "t120.ufs", e.snapshot())
    ahead = _run(e, K)
    fresh = make_engine(generate("single_intersection", demand_rate=800), seed=2)
    loaded = load_state(path)
    fresh.restore(loaded)  # a new engine: nothing is shared with the saved one
    assert fresh.digest() == state_digest(loaded)
    assert fresh.held[0] and fresh.parked
    assert _run(fresh, K) == ahead


def test_restore_across_scenarios_or_configs_raises(make_engine: MakeEngine) -> None:
    scenario = generate("single_intersection", demand_rate=800)
    e = make_engine(scenario)
    _run(e, 20)
    s = e.snapshot()
    with pytest.raises(SimulationError, match="another scenario"):
        make_engine(generate("single_intersection", demand_rate=801)).restore(s)
    with pytest.raises(SimulationError, match=r"differs in: deadlock_timeout"):
        make_engine(scenario, deadlock_timeout=60.0).restore(s)
    other_seed = make_engine(scenario, seed=9)
    with pytest.raises(SimulationError, match="differs in: seed"):
        other_seed.restore(s)
