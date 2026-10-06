"""The signal-control decision task shared by the Gymnasium and PettingZoo envs (plan L).

Every controlled (signalised) intersection is an *agent*. Every ``decision_interval``
seconds each agent picks the phase it wants (``choose_phase``); the engine's signal runtime
enforces min-green, yellow and all-red, so an agent can never produce an unsafe transition.
Actions the signal cannot honour yet (during min-green or a transition) are masked; a masked
action is replaced by the current phase and reported in ``info["action_overridden"]``.

Observations (all in [0, 1]): the current/target phase one-hot, ``min_green_ok``, and per
incoming lane (canonical clockwise order) density ``min(1, n / c)`` and stop-line queue
``min(1, Q / c)``, with ``c = max(1, L / 7.5 m)``.

Rewards (``reward=`` a name or ``{name: weight}``):

* ``diff_waiting_time``: drop in the summed waiting time of vehicles on incoming lanes, /100
* ``queue``: minus the halting vehicles on incoming lanes
* ``pressure``: minus |sum over connectors of n_in/c_in - n_out/c_out| (PressLight)

``terminated`` = demand exhausted and network empty (``Simulation.is_drained``);
``truncated`` = the episode's time limit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from urbanflow.core.constants import VEH_SPACING_REF_M, WAIT_REWARD_SCALE_S
from urbanflow.core.errors import ConfigError
from urbanflow.core.types import BoolArray, FloatArray, IntArray
from urbanflow.scenario.scenario import Scenario
from urbanflow.simulation import Simulation

__all__ = ["REWARDS", "AgentSpec", "SignalControlTask", "load_scenario"]

REWARDS: Final = ("diff_waiting_time", "queue", "pressure")


def load_scenario(scenario: Scenario | str) -> Scenario:
    """A Scenario from an object, a bundled name or a file path."""
    if isinstance(scenario, Scenario):
        return scenario
    from urbanflow.scenario.io import bundled, bundled_names

    return Scenario.load(bundled(scenario) if scenario in bundled_names() else scenario)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Static description of one controlled intersection."""

    id: str
    in_lanes: IntArray
    """Incoming lane link ids, clockwise by approach bearing then lane index."""
    in_capacity: FloatArray
    conn_from: IntArray
    conn_to: IntArray
    phase_conn: BoolArray
    """``[P, C]``: connector ``c`` is green (G or g) in phase ``p``."""
    n_phases: int

    @property
    def obs_size(self) -> int:
        return self.n_phases + 1 + 2 * int(self.in_lanes.size)


def _agents(sim: Simulation, ids: Sequence[str]) -> list[AgentSpec]:
    net = sim.network.compiled
    out = []
    for name in ids:
        j = net.int_index[name]
        lanes = net.int_in_lanes[net.int_in_ptr[j] : net.int_in_ptr[j + 1]].astype(np.intp)
        cap = np.maximum(1.0, net.link_length[lanes] / VEH_SPACING_REF_M)
        phases = sim.signals.phases(name)
        mov_index = {m: k for k, m in enumerate(net.mov_ids)}
        conns: list[int] = []
        owner: list[int] = []
        for m in (i for i, x in enumerate(net.mov_intersection.tolist()) if x == j):
            for c in net.mov_conn[net.mov_conn_ptr[m] : net.mov_conn_ptr[m + 1]].tolist():
                conns.append(c)
                owner.append(m)
        conn = np.asarray(conns, dtype=np.intp)
        green = np.zeros((len(phases), conn.size), dtype=bool)
        for p, ph in enumerate(phases):
            movs = {mov_index[m] for m in ph.green}
            green[p] = [o in movs for o in owner]
        k = conn - net.n_lanes
        out.append(
            AgentSpec(
                id=name,
                in_lanes=lanes,
                in_capacity=cap,
                conn_from=net.conn_from_lane[k].astype(np.intp),
                conn_to=net.conn_to_lane[k].astype(np.intp),
                phase_conn=green,
                n_phases=len(phases),
            )
        )
    return out


class SignalControlTask:
    """One simulation, several signal agents, a fixed decision interval."""

    def __init__(
        self,
        scenario: Scenario | str,
        intersections: Sequence[str] | None = None,
        *,
        decision_interval: float = 5.0,
        reward: str | Mapping[str, float] = "diff_waiting_time",
        episode_length: float | None = None,
        warmup: float = 0.0,
        **overrides: Any,
    ) -> None:
        self.scenario = load_scenario(scenario)
        self.warmup = float(warmup)
        if episode_length is not None:
            overrides["duration"] = self.warmup + float(episode_length)
        self.sim = Simulation(self.scenario, **overrides)
        signalised = list(self.sim.signals.ids)
        if not signalised:
            raise ConfigError(f'scenario "{self.scenario.name}" has no signalised intersection')
        ids = list(intersections) if intersections is not None else signalised
        unknown = sorted(set(ids) - set(signalised))
        if unknown:
            raise ConfigError(
                f"not signalised: {', '.join(unknown)} (signalised: {', '.join(signalised)})"
            )
        self.agents: list[AgentSpec] = _agents(self.sim, ids)
        steps = decision_interval / self.sim.dt
        if decision_interval <= 0 or abs(steps - round(steps)) > 1e-9:
            raise ConfigError(
                f"decision_interval ({decision_interval} s) must be a positive multiple of dt "
                f"({self.sim.dt} s)"
            )
        self.k = round(steps)
        weights = {reward: 1.0} if isinstance(reward, str) else dict(reward)
        bad = sorted(set(weights) - set(REWARDS))
        if bad:
            raise ConfigError(f"unknown reward {', '.join(bad)} (available: {', '.join(REWARDS)})")
        self.reward_weights = weights
        self._prev_wait: dict[str, float] = {}

    @property
    def ids(self) -> list[str]:
        return [a.id for a in self.agents]

    # ------------------------------------------------------------------ episode
    def reset(self, seed: int) -> tuple[dict[str, FloatArray], dict[str, dict[str, Any]]]:
        """Start an episode with ``seed`` (re-seeds all demand)."""
        sim = self.sim
        sim.reset(seed)
        if self.warmup > 0:
            sim.run(until=self.warmup)
        for a in self.agents:
            sim.signals.set_controller(a.id, "external")
        self._prev_wait = {a.id: self._waiting(a) for a in self.agents}
        return self._observe(), self._infos({a.id: False for a in self.agents}, {})

    def action_masks(self) -> dict[str, np.ndarray]:
        """Per agent: which phases can be chosen now (int8, Gymnasium ``sample(mask=)``)."""
        out = {}
        for a in self.agents:
            view = self.sim.signals[a.id]
            mask = np.zeros(a.n_phases, dtype=np.int8)
            current = self._green(a.id)
            if view.stage != "green" or view.green_elapsed < view.min_green - 1e-9:
                mask[current] = 1
            else:
                mask[:] = 1
            out[a.id] = mask
        return out

    def step(
        self, actions: Mapping[str, int]
    ) -> tuple[
        dict[str, FloatArray],
        dict[str, float],
        bool,
        bool,
        dict[str, dict[str, Any]],
    ]:
        """Apply the actions, advance one decision interval; ``(obs, rewards, terminated,
        truncated, infos)`` (termination is shared by all agents)."""
        sim = self.sim
        masks = self.action_masks()
        overridden: dict[str, bool] = {}
        for a in self.agents:
            want = int(actions.get(a.id, self._green(a.id)))
            ok = 0 <= want < a.n_phases and bool(masks[a.id][want])
            overridden[a.id] = not ok
            target = want if ok else self._green(a.id)
            if target != self._green(a.id):
                sim.signals.request_phase(a.id, target)
        for _ in range(self.k):
            if sim.done:
                break
            sim.step()
        components: dict[str, dict[str, float]] = {}
        rewards: dict[str, float] = {}
        for a in self.agents:
            comp = self._reward_components(a)
            components[a.id] = comp
            rewards[a.id] = float(
                sum(self.reward_weights[n] * comp[n] for n in self.reward_weights)
            )
        terminated = sim.is_drained()
        truncated = not terminated and sim.done
        return self._observe(), rewards, terminated, truncated, self._infos(overridden, components)

    # ------------------------------------------------------------------ features
    def _green(self, agent: str) -> int:
        view = self.sim.signals[agent]
        return (
            int(view.target)
            if view.stage != "green" and view.target >= 0
            else int(view.phase_index)
        )

    def _observe(self) -> dict[str, FloatArray]:
        counts = self.sim.lanes.vehicle_counts()
        queues = self.sim.lanes.queue_lengths()
        out = {}
        for a in self.agents:
            view = self.sim.signals[a.id]
            phase = np.zeros(a.n_phases, dtype=np.float32)
            phase[self._green(a.id)] = 1.0
            ok = float(view.stage == "green" and view.green_elapsed >= view.min_green - 1e-9)
            density = np.minimum(1.0, counts[a.in_lanes] / a.in_capacity)
            queue = np.minimum(1.0, queues[a.in_lanes] / a.in_capacity)
            out[a.id] = np.concatenate([phase, [ok], density, queue]).astype(np.float32)
        return out

    def _waiting(self, a: AgentSpec) -> float:
        return float(self.sim.lanes.waiting_times()[a.in_lanes].sum())

    def _reward_components(self, a: AgentSpec) -> dict[str, float]:
        lanes = self.sim.lanes
        out: dict[str, float] = {}
        if "diff_waiting_time" in self.reward_weights:
            now = self._waiting(a)
            out["diff_waiting_time"] = (self._prev_wait[a.id] - now) / WAIT_REWARD_SCALE_S
            self._prev_wait[a.id] = now
        if "queue" in self.reward_weights:
            out["queue"] = -float(lanes.halting_counts()[a.in_lanes].sum())
        if "pressure" in self.reward_weights:
            n = lanes.vehicle_counts()
            cap = np.maximum(1.0, self.sim.network.compiled.link_length / VEH_SPACING_REF_M)
            diff = n[a.conn_from] / cap[a.conn_from] - n[a.conn_to] / cap[a.conn_to]
            out["pressure"] = -abs(float(diff.sum()))
        return out

    def link_counts(self) -> IntArray:
        """Running vehicles per lane (used by baselines such as max-pressure)."""
        return self.sim.lanes.vehicle_counts()

    def _infos(
        self, overridden: Mapping[str, bool], components: Mapping[str, Mapping[str, float]]
    ) -> dict[str, dict[str, Any]]:
        sim = self.sim
        masks = self.action_masks()
        system = {
            "time": sim.time,
            "step": sim.step_count,
            "system_total_running": len(sim.vehicles),
        }
        return {
            a.id: {
                "action_mask": masks[a.id],
                "phase": self._green(a.id),
                "action_overridden": bool(overridden.get(a.id, False)),
                "reward_components": dict(components.get(a.id, {})),
                **system,
            }
            for a in self.agents
        }

    def summary(self) -> dict[str, float]:
        """The episode's metrics summary so far (travel time, delay, waiting ...)."""
        return {k: v for k, v in self.sim.metrics.summary().items() if not math.isnan(v)}
