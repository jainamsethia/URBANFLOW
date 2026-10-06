"""The :class:`Simulation` facade: load, deep-check, compile, build and drive a run (AA 5.3).

Thin composition only: the engine owns the state and the step pipeline; the facade adds
the lifecycle (``done``, ``reset``, ``close``), ``run()`` with progress, per-step caches
of the read views, the signal controllers chosen with ``controllers=`` and the run summary.
Event callbacks (``sim.events``) and snapshots (``snapshot``/``restore``/
``state_digest``) are here too. Control calls made between steps take effect in the next
step (AA 5.5).

After each engine step (F.1): the per-step caches are invalidated first, then the summary
accumulates the step's events, and the event callbacks run last.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import time as wall_clock
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, Unpack

import numpy as np

from urbanflow.checks import deep_check
from urbanflow.control import EventsAPI, SignalsAPI
from urbanflow.core import constants as C
from urbanflow.core.config import ConfigOverrides, SimulationConfig, resolve_config
from urbanflow.core.errors import ConfigError, NotFoundError, SimulationError, suggest
from urbanflow.core.events import EventBus, EventType
from urbanflow.core.rng import RngStreams
from urbanflow.core.types import IntersectionKind
from urbanflow.engine import Engine
from urbanflow.engine.state import check_compatible
from urbanflow.metrics import MetricsManager
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.replay.recorder import ReplayRecorder
from urbanflow.results import SimulationResult, provenance
from urbanflow.routing import Router, router_registry
from urbanflow.scenario.scenario import Scenario
from urbanflow.scenario.schema import ScenarioSpec
from urbanflow.signals import ControllerRef, build_programs
from urbanflow.snapshot import Snapshot
from urbanflow.vehicles import VehicleTypes, car_following_registry
from urbanflow.views import (
    IntersectionCollection,
    LaneCollection,
    NetworkInfo,
    RoadCollection,
    StateArrays,
    StepCache,
    VehicleCollection,
)
from urbanflow.visualization.frames import Frame, LaneMetric, build_frame

__all__ = ["ControllerRef", "MetricsAPI", "ProgressCallback", "ProgressInfo", "Simulation"]

_log = logging.getLogger("urbanflow.simulation")
_ARRIVED = EventType.vehicle_arrived.code
_INSERTED = EventType.vehicle_inserted.code


@dataclass(frozen=True, slots=True)
class ProgressInfo:
    """What a ``run(progress=callback)`` callback receives (about every 0.1 s, and last)."""

    time: float
    end_time: float | None
    """Time at which this ``run()`` stops; None when it runs until the network drains."""
    step_count: int
    running: int
    arrived: int
    steps_per_s: float
    """Steps per wall-clock second in this ``run()`` call."""


ProgressCallback = Callable[[ProgressInfo], None]


MetricsAPI = MetricsManager
"""``sim.metrics`` (the metrics manager; name kept for the public API)."""


def _as_scenario(
    source: Scenario | ScenarioSpec | Mapping[str, Any] | str | os.PathLike[str],
) -> Scenario:
    if isinstance(source, Scenario):
        return source
    if isinstance(source, ScenarioSpec):
        return Scenario.from_spec(source)
    if isinstance(source, Mapping):
        return Scenario.from_dict(source)
    return Scenario.load(source)


def _controller_overrides(
    network: CompiledNetwork, controllers: Mapping[str, ControllerRef]
) -> dict[int, ControllerRef]:
    """``controllers=`` by intersection index: ``"*"`` applies to every signalised
    intersection and per-id entries win. Unknown ids raise ``NotFoundError``; the engine
    rejects unsignalised ones. An instance shared by two intersections is a ``ConfigError``
    (each intersection needs its own controller state)."""
    for key in controllers:
        if key != "*" and key not in network.int_index:
            raise NotFoundError(
                f'controllers: unknown intersection "{key}"{suggest(key, network.int_index)} '
                '(use an intersection id, or "*" for every signalized intersection)'
            )
    out: dict[int, ControllerRef] = {}
    if "*" in controllers:
        signalised = network.int_kind == IntersectionKind.signalized.code
        out = dict.fromkeys(np.flatnonzero(signalised).tolist(), controllers["*"])
    out.update((network.int_index[k], ref) for k, ref in controllers.items() if k != "*")
    instances = [id(r) for r in out.values() if not isinstance(r, type) and hasattr(r, "decide")]
    if len(set(instances)) < len(instances):
        raise ConfigError(
            "controllers: one controller instance cannot run several intersections; "
            "pass a factory instead (e.g. its class), which is called per intersection"
        )
    return out


def _check_supported(config: SimulationConfig) -> None:
    """Config features scheduled for later versions fail by name instead of being ignored."""
    if config.metrics.collectors != ("default",):
        names = ", ".join(config.metrics.collectors)
        raise ConfigError(
            f"custom metric collectors ({names}) are not available in this version; "
            'use metrics.collectors = ["default"]'
        )


class Simulation:
    """One simulation run: scenario + config + engine + read views (plan AA 5.3).

    ``scenario`` is a :class:`Scenario`, a ``ScenarioSpec``, a scenario mapping or a file
    path. The config is resolved as defaults < ``scenario.simulation`` < ``config`` (its set
    fields) < ``router`` < ``**overrides``; ``router`` may also be a :class:`Router`
    instance. ``controllers`` replaces the scenario's signal controllers: keys are
    intersection ids or ``"*"`` (every signalised intersection; per-id entries win), values
    a registry name, ``{"type", "params"}`` or a zero-argument factory returning a
    controller (called again at every ``reset()``). The constructor deep-checks the
    scenario and the named controllers (``urbanflow.check``; errors raise
    ``ScenarioValidationError``), compiles the network, builds the engine and resets to
    ``config.seed``.
    """

    # mypy flags the `router` key of ConfigOverrides as overlapping the explicit parameter;
    # the parameter wins for callers, who are still type-checked against ConfigOverrides.
    def __init__(  # type: ignore[misc]
        self,
        scenario: Scenario | ScenarioSpec | Mapping[str, Any] | str | os.PathLike[str],
        config: SimulationConfig | None = None,
        *,
        controllers: Mapping[str, ControllerRef] | None = None,
        router: str | Router | None = None,
        record: bool | str | os.PathLike[str] | Mapping[str, Any] = False,
        **overrides: Unpack[ConfigOverrides],
    ) -> None:
        self.scenario = _as_scenario(scenario)
        self._recorder: ReplayRecorder | None = None
        if isinstance(record, Mapping):  # a RecordConfig override, e.g. {"enabled": True}
            overrides["record"] = record
            record = False
        name = (
            router
            if router is None or isinstance(router, str)
            else getattr(type(router), "name", type(router).__name__)
        )
        self.config: SimulationConfig = resolve_config(
            ("scenario", self.scenario.spec.simulation),
            ("config", config),
            ("router", None if name is None else {"router": name}),
            ("overrides", dict(overrides)),
        )
        _check_supported(self.config)
        instance = None if router is None or isinstance(router, str) else router
        report, network = deep_check(
            self.scenario, self.config, router=instance, controllers=controllers
        )
        source = str(self.scenario.path) if self.scenario.path else self.scenario.name
        report.raise_for_errors(source=source)
        if network is None:  # pragma: no cover - a network that fails to compile has errors
            raise SimulationError("the network did not compile")
        for warning in report.warnings:
            if warning not in self.scenario.issues:
                _log.warning("%s: %s: %s", source, warning.path, warning.message)
        cfg = self.config
        model = car_following_registry.get(cfg.car_following)
        router_obj: Router = router_registry.get(cfg.router)() if instance is None else instance
        self._engine = Engine(
            network,
            cfg,
            rng=RngStreams(cfg.seed),
            router=router_obj,
            car_following=model(),
            types=VehicleTypes.from_specs(
                network.vehicle_types, model.Params, model=cfg.car_following
            ),
            demand=self.scenario.resolved.demand,
            signals=build_programs(network, self.scenario.resolved.network),
            controllers=_controller_overrides(network, controllers or {}),
        )
        dt, duration = cfg.dt, cfg.duration
        self._n_steps = None if duration is None else math.floor(duration / dt + C.TIME_EPS)
        self._cache: StepCache = {}
        self.metrics = MetricsAPI(self._engine)
        """Metrics: ``summary()``, ``timeseries()``, ``intersections()``, ``trips()``."""
        self.network = NetworkInfo(network, self._engine.graph)
        self.vehicles = VehicleCollection(self._engine, self._cache, self.metrics.arrived_uids)
        self.lanes = LaneCollection(self._engine, self._cache)
        self.roads = RoadCollection(self._engine, self._cache)
        self.state = StateArrays(self._engine, self._cache)
        self.intersections = IntersectionCollection(self._engine, self._cache)
        self.signals = SignalsAPI(self._engine, self._cache)
        """Signal state and control (``sim.signals["J"]``, ``request_phase``, ...)."""
        self._bus = EventBus()
        self.events = EventsAPI(self._bus, self.metrics.event_counts)
        """Event callbacks (``sim.events.subscribe``) and counts since reset."""
        self._closed = False
        self._clear(cfg.seed)
        if record is not False or cfg.record.enabled:
            target = cfg.record.path if record is True or record is False else record
            self.start_recording(target)
        _log.info(
            "simulation initialised",
            extra={
                "scenario": self.scenario.name,
                "seed": cfg.seed,
                "dt": cfg.dt,
                "duration": cfg.duration,
                "accel": cfg.accel,
            },
        )

    @classmethod
    def from_scenario(cls, source: Scenario | str | os.PathLike[str], **kwargs: Any) -> Simulation:
        """Same as ``Simulation(source, **kwargs)`` (e.g. with ``urbanflow.bundled(name)``)."""
        return cls(source, **kwargs)

    # ------------------------------------------------------------------ read-only state
    @property
    def seed(self) -> int:
        """Root seed of the current run (``reset(seed)`` changes it)."""
        return self._seed

    @property
    def dt(self) -> float:
        return self.config.dt

    @property
    def time(self) -> float:
        """Simulated time, s: ``step_count * dt``."""
        return self._engine.time

    @property
    def step_count(self) -> int:
        return self._engine.step_count

    @property
    def end_time(self) -> float | None:
        """``floor(duration / dt) * dt``; None when the run lasts until the network drains."""
        return None if self._n_steps is None else self._n_steps * self.config.dt

    @property
    def done(self) -> bool:
        """The time limit is reached, or (``duration=None``) the network has drained."""
        if self._n_steps is None:
            return self.is_drained()
        return self.step_count >= self._n_steps

    def is_done(self) -> bool:
        """Same as :attr:`done`."""
        return self.done

    def is_drained(self) -> bool:
        """Demand is exhausted and no vehicle is running or waiting (ignores the limit)."""
        eng = self._engine
        return eng.demand_exhausted and not eng.vehicles.active.any()

    # ------------------------------------------------------------------ lifecycle
    def _clear(self, seed: int) -> None:
        self._seed = seed
        self._corrupted = False
        self._interrupted = False
        self._wall = 0.0
        self._cache.clear()
        self.metrics.reset()
        _log.debug("simulation reset", extra={"seed": seed, "dt": self.config.dt})

    def reset(self, seed: int | None = None) -> None:
        """Fresh runtime from the cached network; ``seed=None`` reuses the current seed.

        Re-seeds every random stream, clears vehicles, counters, the summary and the event
        counts, and rebuilds the configured signal controllers (commands such as
        ``set_controller`` and manual holds are undone), so repeated ``reset()`` reproduces
        the run. Event subscriptions are kept.
        """
        if self._closed:
            raise SimulationError("the simulation is closed")
        seed = self._seed if seed is None else seed
        if seed < 0:
            raise ConfigError(f"seed must be >= 0 (got {seed})")
        self._engine.reset(seed)
        self._clear(seed)
        if self._recorder is not None:  # a reset starts the recording over
            path = self._recorder.path
            self._recorder.discard()
            self._recorder = ReplayRecorder(self, path, every=self.config.record.every)

    def _check_steppable(self) -> None:
        if self._closed:
            raise SimulationError("the simulation is closed")
        if self._corrupted:
            raise SimulationError(
                "the simulation state is corrupted (an exception interrupted a step); "
                "call reset() before stepping again"
            )

    def _check_intact(self) -> None:
        if self._corrupted:
            raise SimulationError(
                "the simulation is corrupted (an exception interrupted a step); reset() or "
                "restore() a snapshot taken before it"
            )

    def _step_once(self) -> None:
        start = wall_clock.perf_counter()
        try:
            self._engine.step()
            self._cache.clear()  # first: nothing may read the previous step's objects
            self.metrics.update()
            if self._recorder is not None:
                self._recorder.record()
        except BaseException as exc:  # e.g. Ctrl-C mid-kernel: must reset()
            self._cache.clear()
            self._corrupted = True
            self._interrupted = isinstance(exc, KeyboardInterrupt)
            raise
        finally:
            self._wall += wall_clock.perf_counter() - start
        if self._bus.has_subscribers:  # user callbacks last; the step is complete
            net = self._engine.network
            self._bus.dispatch(
                self._engine.events,
                vehicle_ids=self._engine.vehicles.uid_to_id,
                link_ids=net.link_ids,
                intersection_ids=net.int_ids,
            )

    def step(self, n: int = 1) -> None:
        """Advance ``n`` steps (stopping early once :attr:`done`).

        Raises ``SimulationError`` if the simulation is already done, closed or corrupted.
        """
        self._check_steppable()
        if self.done:
            raise SimulationError(
                f"the simulation is done (t={self.time:g} s); call reset() to run again"
            )
        if n < 1:
            raise SimulationError(f"n must be >= 1 (got {n})")
        for _ in range(n):
            self._step_once()
            if self.done:
                break

    def run(
        self,
        until: float | None = None,
        *,
        duration: float | None = None,
        progress: bool | ProgressCallback = False,
    ) -> SimulationResult:
        """Step until the absolute time ``until``, for ``duration`` more seconds, or (both
        None) until :attr:`done`; never past the time limit. Returns :meth:`get_results`.

        ``progress=True`` draws a progress bar on stderr; a callable receives a
        :class:`ProgressInfo` about every 0.1 s and once at the end.
        """
        self._check_steppable()
        if until is not None and duration is not None:
            raise ConfigError("run() takes until or duration, not both")
        target = until if duration is None else self.time + duration
        if target is not None and not math.isfinite(target):
            raise ConfigError(f"run() needs a finite time (got {target})")
        dt = self.config.dt
        stop = None if target is None else max(self.step_count, math.ceil(target / dt - C.TIME_EPS))
        if self._n_steps is not None:
            stop = self._n_steps if stop is None else min(stop, self._n_steps)
        first, start = self.step_count, wall_clock.perf_counter()

        def info() -> ProgressInfo:
            elapsed = wall_clock.perf_counter() - start
            eng = self._engine
            return ProgressInfo(
                time=self.time,
                end_time=None if stop is None else stop * dt,
                step_count=self.step_count,
                running=int(np.count_nonzero(eng.vehicles.active)),
                arrived=eng.arrived,
                steps_per_s=(self.step_count - first) / elapsed if elapsed > 0 else 0.0,
            )

        try:
            with _reporter(progress, None if stop is None else stop * dt) as report:
                shown = start
                while not self.done and (stop is None or self.step_count < stop):
                    self._step_once()
                    now = wall_clock.perf_counter()
                    if report is not None and now - shown >= C.PROGRESS_REFRESH_S:
                        report(info())
                        shown = now
                if report is not None:
                    report(info())
        except KeyboardInterrupt:  # between steps too (progress, callbacks)
            self._interrupted = True
            raise
        result = self.get_results()
        _log.info(
            "run finished",
            extra={
                "sim_time": result.sim_time,
                "wall_s": round(result.wall_time, 3),
                "steps_per_s": round(info().steps_per_s),
                "arrived": self._engine.arrived,
                "digest": result.state_digest[: C.SHORT_HASH_LENGTH],
            },
        )
        return result

    def close(self) -> None:
        """Release the simulation; later ``step``/``run``/``reset`` raise. Idempotent."""
        if self._recorder is not None and not self._closed:
            self._recorder.close()
        self._closed = True
        self._cache.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ recording
    def start_recording(self, path: str | os.PathLike[str] | None = None) -> None:
        """Record a replay from the current step to ``path`` (default: a ``.ufr`` in the
        workspace ``replays`` directory named after the scenario and seed)."""
        if self._recorder is not None:
            raise SimulationError(f"already recording to {self._recorder.path}")
        if path is None:
            from urbanflow.core.settings import load_settings

            stamp = wall_clock.strftime("%Y%m%dT%H%M%S")
            path = load_settings().replays_dir / f"{self.scenario.name}-s{self._seed}-{stamp}.ufr"
        self._recorder = ReplayRecorder(self, path, every=self.config.record.every)

    def stop_recording(self) -> Path:
        """Finish the recording and pack it; returns the ``.ufr`` path."""
        if self._recorder is None:
            raise SimulationError("not recording (pass record= or call start_recording())")
        path = self._recorder.close()
        self._recorder = None
        return path

    @property
    def recording(self) -> Path | None:
        """Where the current recording goes, or None."""
        return None if self._recorder is None else self._recorder.path

    # ------------------------------------------------------------------ persistence
    def snapshot(self) -> Snapshot:
        """Deep copies of the engine state (vehicles, signals and controllers, demand,
        RNG streams, counters) and of the summary accumulators (F.5).

        ``SimulationError`` if the simulation is corrupted (an exception interrupted a
        step): a half-applied step must not be saved and restored as a good state.
        """
        self._check_intact()
        eng = self._engine
        snap = Snapshot(
            state=eng.snapshot(),
            metrics=self.metrics.state_dict(),
            scenario_hash=self.scenario.content_hash,
            config=self.config,
            time=self.time,
        )
        _log.debug("snapshot taken", extra={"step": eng.step_count})
        return snap

    def restore(self, snapshot: Snapshot) -> None:
        """Continue from ``snapshot``: the following steps equal those after it was taken.

        ``SimulationError`` if it was taken from another scenario, with another config or
        with other configured signal controllers (``controllers=``).
        Clears the corrupted flag and every per-step cache. A snapshot loaded from a file
        has no summary accumulators: they restart empty (a WARNING is logged).
        """
        if self._closed:
            raise SimulationError("the simulation is closed")
        eng = self._engine
        check_compatible(eng, snapshot.state)
        self._cache.clear()
        try:
            eng.restore(snapshot.state)
        except BaseException:
            self._corrupted = True
            raise
        if snapshot.metrics is None:
            self.metrics.reset()
            _log.warning(
                "restored a snapshot loaded from a file (t=%g s): the run summary (travel "
                "times, event counts, the ids of arrived vehicles) restarts empty",
                snapshot.time,
            )
        else:
            self.metrics.load_state_dict(snapshot.metrics)
        self._seed = eng.rng.seed
        self._corrupted = False
        self._interrupted = False
        _log.debug("snapshot restored", extra={"step": eng.step_count})

    def frame(self, *, lane_metric: int = 0, signals: bool = True) -> Frame:
        """The render :class:`~urbanflow.visualization.Frame` of the current step
        (``lane_metric`` is a :class:`~urbanflow.visualization.LaneMetric` code)."""
        self._check_intact()
        return build_frame(self._engine, signals=signals, lane_metric=LaneMetric(lane_metric))

    def vehicle_meta(self, uids: Iterable[int]) -> tuple[list[str], list[int], list[int]]:
        """``(ids, type indices, destination road indices)`` of vehicles by uid (B.2 #23);
        unknown or finished vehicles get ``("", -1, -1)``."""
        eng = self._engine
        veh, routes = eng.vehicles, eng.routes.routes
        ids, types, dests = [], [], []
        for uid in uids:
            vid = veh.uid_to_id[uid] if 0 <= uid < len(veh.uid_to_id) else ""
            h = veh.id_to_handle.get(vid, -1)
            ids.append(vid)
            types.append(int(veh.type_idx[h]) if h >= 0 else -1)
            dests.append(int(routes[int(veh.route_id[h])][-1]) if h >= 0 else -1)
        return ids, types, dests

    def state_digest(self) -> str:
        """sha256 hex of the canonical engine state (F.5): on the same platform, equal
        digests mean equal states and equal futures. ``SimulationError`` if the simulation
        is corrupted (an exception interrupted a step)."""
        self._check_intact()
        return self._engine.digest()

    # ------------------------------------------------------------------ results
    def get_results(self) -> SimulationResult:
        """Results so far (partial if not done; ``interrupted`` after Ctrl-C)."""
        return SimulationResult(
            scenario_name=self.scenario.name,
            scenario_hash=self.scenario.content_hash,
            config=self.config,
            seed=self._seed,
            sim_time=self.time,
            steps=self.step_count,
            wall_time=self._wall,
            interrupted=self._interrupted,
            summary=self.metrics.summary(),
            tables=self.metrics.tables(),
            event_counts=self.metrics.event_counts(),
            provenance=provenance(),
            controllers={
                self._engine.network.int_ids[j]: spec for j, spec in self._engine.configured.items()
            },
            state_digest="" if self._corrupted else self.state_digest(),
        )

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"t={self.time:g}s"
        return f"Simulation({self.scenario.name!r}, seed={self._seed}, {state})"


@contextlib.contextmanager
def _reporter(
    progress: bool | ProgressCallback, total: float | None
) -> Iterator[ProgressCallback | None]:
    """The progress sink of one ``run()``: a callback, a rich bar on stderr, or None."""
    if progress is False:
        yield None
        return
    if callable(progress):
        yield progress
        return
    from rich.console import Console
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

    columns = (
        TextColumn("Running"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[info]}"),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=Console(stderr=True)) as bar:
        task = bar.add_task("run", total=total, info="")

        def update(p: ProgressInfo) -> None:
            end = "" if p.end_time is None else f"/{p.end_time:g}"
            text = f"t={p.time:g}{end} s  {p.running} veh  {p.steps_per_s:,.0f} steps/s"
            bar.update(task, completed=p.time, info=text)

        yield update
