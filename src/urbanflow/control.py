"""Control namespaces of a simulation: ``sim.signals`` and ``sim.events`` (plan AA 5.4,
H.2, F.4, F.8).

Reads return snapshot copies (:class:`~urbanflow.views.SignalView`,
:class:`~urbanflow.views.PhaseInfo`) or read-only arrays cached for the current step.
Commands are the engine's (``engine.commands``): they validate their arguments, are
logged, and take effect in the next step. Intersections are given by id (or global
index), phases by index or phase id; indices may be Python or numpy integers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING

import numpy as np

from urbanflow.core.constants import TIME_EPS
from urbanflow.core.errors import CommandError, NotFoundError
from urbanflow.core.events import EventBus, EventCallback, EventType, Subscription
from urbanflow.core.types import IntArray, Stage, UIntArray
from urbanflow.signals import ControllerRef, SignalController
from urbanflow.views import (
    PhaseInfo,
    SignalView,
    StepCache,
    _cached,
    phase_infos,
    signal_view,
)

if TYPE_CHECKING:
    from urbanflow.engine import Engine

__all__ = ["EventsAPI", "SignalsAPI", "Subscription"]


class SignalsAPI:
    """``sim.signals``: the state and control of every signalised intersection (AA 5.4).

    ``sim.signals["J"]`` is a :class:`~urbanflow.views.SignalView`. Unknown or
    unsignalised intersections raise ``NotFoundError`` with a hint; invalid commands
    raise ``CommandError`` (AA 5.5).
    """

    def __init__(self, engine: Engine, cache: StepCache) -> None:
        self._engine = engine
        self._cache = cache
        net = engine.network
        self.ids: tuple[str, ...] = tuple(
            net.int_ids[p.intersection] for p in engine.signals.programs
        )
        """Signalised intersection ids, in compiled order (the order of the vectors)."""
        self._rows = np.array([p.intersection for p in engine.signals.programs], dtype=np.intp)

    # ------------------------------------------------------------------ reading
    def _index(self, intersection: object) -> int:
        """The commands' resolver (numpy integers accepted); every failure of a lookup is a
        ``NotFoundError`` here."""
        try:
            return self._engine.commands.signal_index(intersection)
        except CommandError as exc:
            raise NotFoundError(str(exc)) from None

    def __getitem__(self, intersection: str | int) -> SignalView:
        """The current signal state of ``intersection``."""
        return signal_view(self._engine, self._index(intersection))

    def __contains__(self, intersection: object) -> bool:
        """``intersection`` (an id or index) names a signalised intersection."""
        try:
            self._index(intersection)
        except NotFoundError:
            return False
        return True

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[SignalView]:
        for j in self._rows.tolist():
            yield signal_view(self._engine, j)

    def phases(self, intersection: str | int) -> tuple[PhaseInfo, ...]:
        """The phases of the intersection's program, in program order."""
        j = self._index(intersection)
        return phase_infos(self._engine.network, self._engine.signals.program(j))

    def controller(self, intersection: str | int) -> SignalController:
        """The active controller instance (a fresh ``external`` during a manual hold)."""
        return self._engine.controllers[self._index(intersection)]

    def can_switch(self, intersection: str | int) -> bool:
        """A phase request would start its transition in the next step: the signal is in
        green and the phase has had its ``min_green``."""
        j = self._index(intersection)
        sig = self._engine.signals
        min_green = float(sig.program(j).min_green[int(sig.phase[j])])
        return bool(
            sig.stage[j] == Stage.green.code and sig.green_elapsed[j] >= min_green - TIME_EPS
        )

    def phase_indices(self) -> IntArray:
        """Current phase of each signalised intersection, aligned with :attr:`ids`
        (read-only; the phase being left during a transition)."""

        def build() -> IntArray:
            out = self._engine.signals.phase[self._rows].astype(np.int64)
            out.setflags(write=False)
            return out

        return _cached(self._cache, "signal_phases", build)

    def movement_states(self) -> UIntArray:
        """Signal state code of every movement (``sim.network.movement_ids`` order):
        r 0, y 1, g 2, G 3, and 255 at unsignalised intersections (read-only)."""

        def build() -> UIntArray:
            out = self._engine.signals.movement_state.copy()
            out.setflags(write=False)
            return out

        return _cached(self._cache, "signal_states", build)

    # ------------------------------------------------------------------ control
    def request_phase(self, intersection: str | int, phase: int | str) -> None:
        """Ask for ``phase``; min-green, yellow and all-red are honoured. Needs a controller
        that accepts requests (``external`` or a custom one); otherwise ``CommandError``
        explains ``set_controller`` and ``hold_phase``."""
        self._engine.commands.request_phase(intersection, phase)
        self._cache.clear()

    def hold_phase(self, intersection: str | int, phase: int | str) -> None:
        """Manual override with any controller: park it, install a fresh ``external`` and
        request ``phase`` (min-green, yellow and all-red honoured). Calling it again during
        the hold changes the held phase; :meth:`release` ends it."""
        self._engine.commands.hold_phase(intersection, phase)
        self._cache.clear()

    def release(self, intersection: str | int) -> None:
        """End a manual hold: the parked controller continues from the current phase
        (fixed-time offsets may drift). ``CommandError`` if no hold is active."""
        self._engine.commands.release(intersection)
        self._cache.clear()

    def set_phase(self, intersection: str | int, phase: int | str) -> None:
        """Jump to ``phase`` now, without yellow or all-red (setup and tests); the next step
        emits a forced ``phase_changed``."""
        self._engine.commands.set_phase(intersection, phase)
        self._cache.clear()

    def set_controller(self, intersection: str | int, ref: ControllerRef) -> None:
        """Install a controller (a registry name, ``{"type", "params"}``, an instance or a
        zero-argument factory); it continues from the current phase. During a manual hold
        it replaces the parked controller. ``reset()`` restores the configured ones."""
        self._engine.commands.set_controller(intersection, ref)
        self._cache.clear()


class EventsAPI:
    """``sim.events``: callbacks for engine events and cumulative counts (AA 5.4, F.8).

    Callbacks receive frozen :class:`~urbanflow.core.events.Event` objects (ids, never
    engine indices) after each step has completed, in event order; control calls made
    from a callback take effect in the next step, and a callback's exception propagates
    out of ``step()`` with the state already consistent (AA 5.5). Events of types nobody
    subscribed to are never materialised. Subscriptions survive ``reset()``.
    """

    def __init__(self, bus: EventBus, counts: Callable[[], Mapping[str, int]]) -> None:
        self._bus = bus
        self._counts = counts

    def subscribe(
        self,
        types: EventType | str | Iterable[EventType | str] | None,
        callback: EventCallback,
    ) -> Subscription:
        """Call ``callback(event)`` for every event of ``types`` (a type, several, or None
        for all); ``.unsubscribe()`` on the result stops it."""
        return self._bus.subscribe(types, callback)

    def counts(self) -> dict[EventType, int]:
        """Events per type, cumulative since the last reset (every type, zeros included)."""
        return {EventType(name): n for name, n in self._counts().items()}
