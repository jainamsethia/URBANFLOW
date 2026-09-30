"""Signal programs compiled from the resolved ``SignalSpec`` of each signalised node (plan H.1).

A :class:`SignalProgram` holds the phases of one intersection as arrays over its movements
(global movement ids in canonical order, i.e. the compiled order of the intersection's
movements): ``phase_state[p, k]`` is the :class:`~urbanflow.core.types.SignalState` code of
movement ``k`` in phase ``p`` (G 3, g 2, r 0). Yellow and all-red are never listed in
phases; :meth:`SignalProgram.transition` plans them:

* ``losing`` = green (G or g) in ``p`` and red in ``q``; ``gaining`` = red in ``p`` and green
  in ``q``; ``common`` = green in both;
* nothing losing: the switch is immediate;
* otherwise **yellow** for ``yellow`` s (losing = y), then **all-red** for ``all_red`` s
  (losing = r, gaining still r), then green ``q``. Common movements stay green throughout
  and take their ``q`` state (G <-> g) at the start of the transition.

W301 (never green) and W302 (conflicting protected pair) are reported by validation and the
compiler (E.8); programs assume a validated scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from urbanflow.core.errors import NotFoundError, SimulationError, suggest
from urbanflow.core.types import BoolArray, FloatArray, IntArray, IntersectionKind, SignalState
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.network.conflicts import ConflictKind
from urbanflow.scenario.schema import ControllerSpec, NetworkSpec, SignalSpec

__all__ = ["SignalProgram", "Transition", "build_programs"]

StateArray = NDArray[np.uint8]
"""Signal state codes (``SignalState.code``: r 0, y 1, g 2, G 3)."""

_R, _Y, _G = SignalState.r.code, SignalState.y.code, SignalState.g.code


@dataclass(frozen=True, slots=True, eq=False)
class Transition:
    """The plan of ``trans(p -> q)``; arrays are over the program's movements."""

    losing: BoolArray
    """Green in ``p``, red in ``q``: yellow, then red during all-red."""
    gaining: BoolArray
    """Red in ``p``, green in ``q``: red until green ``q`` starts."""
    common: BoolArray
    """Green in both: stays green with its ``q`` state from the start of the transition."""
    yellow: StateArray
    """Movement states during YELLOW(p -> q)."""
    all_red: StateArray
    """Movement states during ALL_RED(p -> q)."""

    @property
    def immediate(self) -> bool:
        """Nothing loses green: green ``q`` starts at once, without yellow or all-red."""
        return not bool(self.losing.any())


@dataclass(frozen=True, slots=True, eq=False)
class SignalProgram:
    """Phases, timing and movement conflicts of one signalised intersection (H.1)."""

    intersection: int
    """Global intersection index."""
    movements: IntArray
    """Global movement ids in canonical order (``M_j`` entries)."""
    phase_ids: tuple[str, ...]
    phase_state: StateArray
    """``[P, M_j]`` state codes: G 3, g 2, r 0."""
    duration: FloatArray
    """Fixed-time green per phase, s."""
    min_green: FloatArray
    max_green: FloatArray
    yellow: float
    all_red: float
    initial_phase: int
    conflict_mm: BoolArray
    """``[M_j, M_j]``: some connector pair of the two movements crosses or merges."""
    controller: ControllerSpec
    """The configured controller (registry name and parameters)."""

    def __post_init__(self) -> None:
        arrays: tuple[NDArray[Any], ...] = (
            self.movements,
            self.phase_state,
            self.duration,
            self.min_green,
            self.max_green,
            self.conflict_mm,
        )
        for arr in arrays:
            arr.setflags(write=False)

    @classmethod
    def from_spec(cls, net: CompiledNetwork, intersection: int, spec: SignalSpec) -> SignalProgram:
        """Compile the resolved ``spec`` of ``intersection`` (a global index) on ``net``."""
        j = intersection
        movements = np.flatnonzero(net.mov_intersection == j)
        local = {net.mov_ids[m]: k for k, m in enumerate(movements.tolist())}
        phases = spec.phases or ()
        if not phases:
            raise SimulationError(f'signal of intersection "{net.int_ids[j]}" has no phases')
        state = np.zeros((len(phases), movements.size), dtype=np.uint8)
        for p, phase in enumerate(phases):
            for mid, s in phase.green.items():
                if mid not in local:
                    raise NotFoundError(
                        f'phase {p} of intersection "{net.int_ids[j]}" lists unknown movement '
                        f'"{mid}"{suggest(mid, local)}'
                    )
                state[p, local[mid]] = SignalState(s).code

        def per_phase(attr: str, default: float) -> FloatArray:
            values = [getattr(ph, attr) for ph in phases]
            return np.array([default if v is None else v for v in values], dtype=np.float64)

        return cls(
            intersection=j,
            movements=movements,
            phase_ids=tuple(ph.id or f"p{k}" for k, ph in enumerate(phases)),
            phase_state=state,
            duration=np.array([ph.duration for ph in phases], dtype=np.float64),
            min_green=per_phase("min_green", spec.min_green),
            max_green=per_phase("max_green", spec.max_green),
            yellow=float(spec.yellow),
            all_red=float(spec.all_red),
            initial_phase=int(spec.initial_phase),
            conflict_mm=_conflict_matrix(net, movements),
            controller=spec.controller,
        )

    # ------------------------------------------------------------------ queries
    @property
    def n_phases(self) -> int:
        return int(self.phase_state.shape[0])

    @property
    def n_movements(self) -> int:
        return int(self.movements.size)

    def phase_index(self, phase: int | str) -> int:
        """Index of ``phase`` (an index or a phase id); ``NotFoundError`` with a hint."""
        if isinstance(phase, str):
            if phase not in self.phase_ids:
                raise NotFoundError(f'unknown phase "{phase}"{suggest(phase, self.phase_ids)}')
            return self.phase_ids.index(phase)
        if not 0 <= phase < self.n_phases:
            raise NotFoundError(
                f"phase {phase} does not exist ({self.n_phases} phases, 0-{self.n_phases - 1})"
            )
        return int(phase)

    def transition(self, p: int, q: int) -> Transition:
        """The yellow / all-red plan of switching from phase ``p`` to phase ``q`` (H.1)."""
        a, b = self.phase_state[p], self.phase_state[q]
        green_a, green_b = a >= _G, b >= _G
        losing, common = green_a & ~green_b, green_a & green_b
        all_red: StateArray = np.where(common, b, _R).astype(np.uint8)
        yellow: StateArray = np.where(losing, _Y, all_red).astype(np.uint8)
        return Transition(losing, ~green_a & green_b, common, yellow, all_red)


def _conflict_matrix(net: CompiledNetwork, movements: IntArray) -> BoolArray:
    """``conflict_mm`` of the movements: any crossing or merging connector pair."""
    local = np.full(net.n_movements, -1, dtype=np.intp)
    local[movements] = np.arange(movements.size)
    kinds = np.isin(net.conf_kind, [int(ConflictKind.crossing), int(ConflictKind.merging)])
    ma = local[net.link_movement[net.conf_a[kinds]]]
    mb = local[net.link_movement[net.conf_b[kinds]]]
    ours = (ma >= 0) & (mb >= 0)
    out = np.zeros((movements.size, movements.size), dtype=bool)
    out[ma[ours], mb[ours]] = True
    out[mb[ours], ma[ours]] = True
    return out


def build_programs(net: CompiledNetwork, network: NetworkSpec) -> tuple[SignalProgram, ...]:
    """The programs of every signalised intersection of the resolved ``network``, in
    ``int_program`` order (the order of the signalised intersections)."""
    programs: list[SignalProgram] = []
    for j, ix in enumerate(network.intersections):
        if net.int_kind[j] != IntersectionKind.signalized.code:
            continue
        if ix.signal is None:
            raise SimulationError(f'signalized intersection "{ix.id}" has no resolved signal')
        programs.append(SignalProgram.from_spec(net, j, ix.signal))
    return tuple(programs)
