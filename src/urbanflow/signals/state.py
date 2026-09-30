"""The signal runtime state machine shared by every controller (plan H.2).

Per signalised intersection the stage is ``GREEN(p)``, ``YELLOW(p -> q)`` or
``ALL_RED(p -> q)``::

    GREEN(p)  --request(q != p) & green_elapsed >= min_green[p] - 1e-9-->  YELLOW(p -> q)
              [or GREEN(q) at once when nothing loses green]
    YELLOW    --stage_elapsed >= yellow - 1e-9-->   ALL_RED   [or GREEN(q) if all_red == 0]
    ALL_RED   --stage_elapsed >= all_red - 1e-9-->  GREEN(q)  (phase_changed, aux = q)

:meth:`SignalRuntime.advance` first applies transitions using the elapsed times of the
previous steps (a pending request, then timed stage ends; zero-length stages are skipped
within the same call), then adds ``dt`` to ``stage_elapsed`` and ``green_elapsed``. A
stage of duration ``d`` is therefore applied in exactly ``ceil(d/dt - 1e-9)`` steps, and
controllers see how long the current phase has already been applied. ``green_elapsed``
restarts at 0 when a green starts and keeps counting through the following transition.

Requests are queued, the last one wins, and a request made during a transition applies
after the new green starts, still honouring its ``min_green``. :meth:`SignalRuntime.force`
(``set_phase``) jumps to ``GREEN(q)`` between steps without intergreen.

**Events.** ``phase_changed`` is emitted when a green starts, with ``intersection = j``,
``link = -1`` and ``aux = q``; a forced ``set_phase`` jump (emitted in the next step, at
its start time) also sets :data:`~urbanflow.core.events.PHASE_FORCED_BIT` in ``aux``
(:func:`~urbanflow.core.events.phase_aux`, B.2 #25).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from urbanflow.core.constants import TIME_EPS
from urbanflow.core.errors import NotFoundError
from urbanflow.core.events import EventBuffer, EventType, phase_aux
from urbanflow.core.types import (
    SIGNAL_CODE_UNSIGNALISED,
    BoolArray,
    SignalState,
    Stage,
)
from urbanflow.signals.program import SignalProgram, StateArray

__all__ = ["SignalRuntime", "SignalSnapshot"]

_GREEN, _YELLOW, _ALL_RED = Stage.green.code, Stage.yellow.code, Stage.all_red.code
_CHARS = "".join(s.value for s in SignalState)  # code -> "r", "y", "g", "G"


@dataclass(frozen=True, slots=True, eq=False)
class SignalSnapshot:
    """Array form of the signal state (H.2); per-intersection arrays have one entry per
    intersection (unsignalised: phase/green/target -1, stage 0, timers NaN)."""

    movement_state: StateArray
    """Per movement: r 0, y 1, g 2, G 3; 255 at unsignalised intersections."""
    phase: NDArray[np.int16]
    """Current phase (the one being left during a transition)."""
    green: NDArray[np.int16]
    """The phase that is or will next be green: ``phase``, or the target in a transition."""
    target: NDArray[np.int16]
    """Transition target, -1 in green."""
    stage: NDArray[np.uint8]
    """``Stage`` code: 0 green, 1 yellow, 2 all_red."""
    stage_elapsed: NDArray[np.float32]
    green_elapsed: NDArray[np.float32]
    min_green: NDArray[np.float32]
    """Of the current phase."""
    max_green: NDArray[np.float32]


class SignalRuntime:
    """Stage timers and movement states of every signalised intersection (H.2).

    ``programs`` are the signalised intersections' programs; intersections and movements
    are addressed by their global indices (``n_intersections``, ``n_movements`` in total).
    """

    def __init__(
        self, programs: Sequence[SignalProgram], *, n_intersections: int, n_movements: int
    ) -> None:
        self.programs: tuple[SignalProgram, ...] = tuple(
            sorted(programs, key=lambda p: p.intersection)
        )
        self._by_int: dict[int, SignalProgram] = {p.intersection: p for p in self.programs}
        n = n_intersections
        self.movement_state: StateArray = np.full(
            n_movements, SIGNAL_CODE_UNSIGNALISED, dtype=np.uint8
        )
        """Current state of every movement (the state applied during this step's move)."""
        self.phase = np.full(n, -1, dtype=np.int16)
        self.target = np.full(n, -1, dtype=np.int16)
        self.pending = np.full(n, -1, dtype=np.int16)
        """Queued request per intersection, -1 if none."""
        self.stage = np.zeros(n, dtype=np.uint8)
        self.stage_elapsed = np.zeros(n, dtype=np.float64)
        self.green_elapsed = np.zeros(n, dtype=np.float64)
        self.yellow_zero = np.zeros(n, dtype=bool)
        """Intersections whose program has ``yellow == 0`` (I10 exemption)."""
        self.last_state: StateArray = self.movement_state.copy()
        """Movement states at the end of the previous step (:meth:`end_step`; I10)."""
        self.forced: BoolArray = np.zeros(n, dtype=bool)
        """Intersections with a forced jump since the previous step's end (I10)."""
        self._forced_events: list[tuple[int, int]] = []
        for prog in self.programs:
            self.yellow_zero[prog.intersection] = prog.yellow <= TIME_EPS
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Every intersection in ``GREEN(initial_phase)`` with zero timers, no requests."""
        self.target[:] = -1
        self.pending[:] = -1
        self.stage[:] = _GREEN
        self.stage_elapsed[:] = 0.0
        self.green_elapsed[:] = 0.0
        self.forced[:] = False
        self._forced_events.clear()
        for prog in self.programs:
            j = prog.intersection
            self.phase[j] = prog.initial_phase
            self.movement_state[prog.movements] = prog.phase_state[prog.initial_phase]
        self.end_step()

    def end_step(self) -> None:
        """Remember the states for the I10 check of the next step (engine, step end)."""
        self.last_state[:] = self.movement_state
        self.forced[:] = False

    # ------------------------------------------------------------------ queries
    def program(self, j: int) -> SignalProgram:
        """The program of intersection ``j``; ``NotFoundError`` if it is not signalised."""
        prog = self._by_int.get(j)
        if prog is None:
            raise NotFoundError(f"intersection {j} is not signalised")
        return prog

    def state_string(self, j: int) -> str:
        """SUMO-style state of intersection ``j``: one of ``rygG`` per movement."""
        codes = self.movement_state[self.program(j).movements].tolist()
        return "".join(_CHARS[c] for c in codes)

    def snapshot(self) -> SignalSnapshot:
        """Copies of the current state in the H.2 array form."""
        n = self.phase.size
        min_green = np.full(n, np.nan, dtype=np.float32)
        max_green = np.full(n, np.nan, dtype=np.float32)
        stage_elapsed = np.full(n, np.nan, dtype=np.float32)
        green_elapsed = np.full(n, np.nan, dtype=np.float32)
        for prog in self.programs:
            j, p = prog.intersection, int(self.phase[prog.intersection])
            min_green[j], max_green[j] = prog.min_green[p], prog.max_green[p]
            stage_elapsed[j], green_elapsed[j] = self.stage_elapsed[j], self.green_elapsed[j]
        in_transition = self.target >= 0
        return SignalSnapshot(
            movement_state=self.movement_state.copy(),
            phase=self.phase.copy(),
            green=np.where(in_transition, self.target, self.phase).astype(np.int16),
            target=self.target.copy(),
            stage=self.stage.copy(),
            stage_elapsed=stage_elapsed,
            green_elapsed=green_elapsed,
            min_green=min_green,
            max_green=max_green,
        )

    # ------------------------------------------------------------------ control
    def request(self, j: int, q: int) -> None:
        """Queue a switch of ``j`` to phase ``q`` (the last request wins)."""
        self.pending[j] = self._checked(j, q)

    def force(self, j: int, q: int) -> None:
        """Jump ``j`` to ``GREEN(q)`` now, without intergreen (``set_phase``); the forced
        ``phase_changed`` event is emitted by the next :meth:`advance`."""
        prog = self.program(j)
        self._start_green(prog, self._checked(j, q))
        self.pending[j] = -1
        self.forced[j] = True
        self._forced_events.append((j, q))

    def place(
        self,
        j: int,
        phase: int,
        *,
        stage: Stage = Stage.green,
        target: int = -1,
        stage_elapsed: float = 0.0,
        green_elapsed: float | None = None,
    ) -> None:
        """Set the state of ``j`` at reset (controllers' initial state, e.g. offsets).

        ``target`` is required in yellow and all-red; ``green_elapsed`` defaults to
        ``stage_elapsed``.
        """
        prog = self.program(j)
        phase = self._checked(j, phase)
        if stage is Stage.green:
            self._start_green(prog, phase)
        else:
            tr = prog.transition(phase, self._checked(j, target))
            if tr.immediate:
                raise ValueError(f"phase {phase} -> {target} has no {stage.value} stage")
            self.phase[j], self.target[j], self.stage[j] = phase, target, stage.code
            states = tr.yellow if stage is Stage.yellow else tr.all_red
            self.movement_state[prog.movements] = states
        self.pending[j] = -1
        self.stage_elapsed[j] = stage_elapsed
        self.green_elapsed[j] = stage_elapsed if green_elapsed is None else green_elapsed

    def advance(self, dt: float, events: EventBuffer, *, step: int, time: float) -> None:
        """One step of every intersection (see the module docstring for the order)."""
        for j, q in self._forced_events:
            aux = phase_aux(q, forced=True)
            events.append(EventType.phase_changed, step, time, intersection=j, aux=aux)
        self._forced_events.clear()
        stage, pending = self.stage, self.pending
        for prog in self.programs:
            j = prog.intersection
            while True:
                if stage[j] == _GREEN:
                    q, p = int(pending[j]), int(self.phase[j])
                    if q < 0:
                        break
                    if q == p:
                        pending[j] = -1
                        break
                    if self.green_elapsed[j] < prog.min_green[p] - TIME_EPS:
                        break
                    pending[j] = -1
                    tr = prog.transition(p, q)
                    if tr.immediate:
                        self._start_green(prog, q, events, step, time)
                        continue
                    self.target[j], stage[j], self.stage_elapsed[j] = q, _YELLOW, 0.0
                    self.movement_state[prog.movements] = tr.yellow
                elif stage[j] == _YELLOW:
                    if self.stage_elapsed[j] < prog.yellow - TIME_EPS:
                        break
                    tr = prog.transition(int(self.phase[j]), int(self.target[j]))
                    stage[j], self.stage_elapsed[j] = _ALL_RED, 0.0
                    self.movement_state[prog.movements] = tr.all_red
                else:
                    if self.stage_elapsed[j] < prog.all_red - TIME_EPS:
                        break
                    self._start_green(prog, int(self.target[j]), events, step, time)
            self.stage_elapsed[j] += dt
            self.green_elapsed[j] += dt

    # ------------------------------------------------------------------ helpers
    def _checked(self, j: int, q: int) -> int:
        n = self.program(j).n_phases
        if not 0 <= q < n:
            raise NotFoundError(f"phase {q} does not exist at intersection {j} ({n} phases)")
        return int(q)

    def _start_green(
        self,
        prog: SignalProgram,
        q: int,
        events: EventBuffer | None = None,
        step: int = 0,
        time: float = 0.0,
    ) -> None:
        j = prog.intersection
        self.phase[j], self.target[j], self.stage[j] = q, -1, _GREEN
        self.stage_elapsed[j] = self.green_elapsed[j] = 0.0
        self.movement_state[prog.movements] = prog.phase_state[q]
        if events is not None:
            events.append(EventType.phase_changed, step, time, intersection=j, aux=phase_aux(q))
