"""Car-following models, the safe-speed cap and the ballistic update (plan G.1-G.3, G.6).

Everything here is vectorised over the evaluated vehicles. The engine calls a model's
``acceleration`` for the leader and for the obstacle, then *always* applies
:func:`safe_speed` and :func:`ballistic`, so a buggy custom model cannot cause collisions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from urbanflow.core import constants as C
from urbanflow.core.registry import Registry
from urbanflow.core.types import BoolArray, FloatArray, IntArray, UIntArray
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.vehicles.table import VehicleTable
from urbanflow.vehicles.types import ParamArrays, VehicleTypes

__all__ = [
    "IDM",
    "CarFollowingInputs",
    "CarFollowingModel",
    "IDMParams",
    "ParamArrays",
    "ballistic",
    "car_following_registry",
    "desired_speed",
    "no_overshoot",
    "register_car_following",
    "safe_speed",
    "stop_budget",
]


@dataclass(frozen=True, slots=True)
class CarFollowingInputs:
    """Inputs of one ``acceleration`` call, arrays over the evaluated vehicles (G.6)."""

    speed: FloatArray
    """Own speed v, m/s."""
    v0: FloatArray
    """Desired speed, m/s."""
    gap: FloatArray
    """Bumper-to-bumper gap to the leader (or obstacle), m; ``inf`` = no leader."""
    leader_speed: FloatArray
    """Leader speed, m/s (0 for obstacles; ignored where ``gap`` is ``inf``)."""
    dt: float
    uid: UIntArray
    rng: np.random.Generator
    """The model's own stream ``model:{name}`` (stochastic models only)."""


class CarFollowingModel(Protocol):
    """A car-following model (G.6); register with ``@register_car_following("name")``."""

    name: ClassVar[str]
    Params: ClassVar[type[BaseModel]]
    """Validates ``vehicle_types[].model_params``; its fields become :class:`ParamArrays`."""

    def acceleration(self, x: CarFollowingInputs, p: ParamArrays) -> FloatArray:
        """Desired acceleration per vehicle, m/s^2 (the engine clamps and caps it)."""
        ...


car_following_registry: Registry[type[CarFollowingModel]] = Registry("car_following")


def register_car_following[M: type[CarFollowingModel]](name: str) -> Callable[[M], M]:
    """Class decorator registering a :class:`CarFollowingModel` as ``name``.

    Sets ``cls.name`` when the class does not define it.
    """

    def decorator(cls: M) -> M:
        if "name" not in vars(cls):
            cls.name = name
        car_following_registry.register(name, cls)
        return cls

    return decorator


class IDMParams(BaseModel):
    """IDM ``model_params``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    delta: float = Field(default=C.IDM_DELTA, gt=0)
    """Free-road acceleration exponent."""


@register_car_following("idm")
class IDM:
    """Intelligent Driver Model with the IIDM-bounded free term (G.1).

    Treiber, Hennecke & Helbing (2000), Phys. Rev. E 62; Treiber & Kesting, *Traffic Flow
    Dynamics* (2013).
    """

    name: ClassVar[str] = "idm"
    Params: ClassVar[type[BaseModel]] = IDMParams

    def acceleration(self, x: CarFollowingInputs, p: ParamArrays) -> FloatArray:
        """``a[max(1 - (v/v0)^delta, -b/a) - (s*/s)^2]`` clamped to ``[-b_emerg, a]``.

        ``s* = s0 + max(0, vT + v dv / (2 sqrt(ab)))``; ``s = inf`` keeps the free term only;
        ``s <= 0.01`` m gives ``-b_emerg``.
        """
        v, gap = x.speed, x.gap
        a, b = p.a, p.b
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # fmax: v = v0 = 0 (a set_speed of 0) gives NaN, which falls back to -b/a
            free = np.fmax(1.0 - (v / x.v0) ** p.delta, -b / a)
            dv = v - x.leader_speed
            s_star = p.s0 + np.maximum(0.0, v * p.T + v * dv / (2 * np.sqrt(a * b)))
            interaction = np.where(np.isinf(gap), 0.0, (s_star / gap) ** 2)
        acc = np.where(gap <= C.IDM_OVERLAP_GAP, -p.b_emerg, a * (free - interaction))
        return np.clip(acc, -p.b_emerg, a)


def no_overshoot(accel: FloatArray, v: FloatArray, v0: FloatArray, dt: float) -> FloatArray:
    """G.1: a positive acceleration never takes v past ``max(v, v0)`` within one step."""
    cap = (np.maximum(v, v0) - v) / dt
    return np.where(accel > 0, np.minimum(accel, cap), accel)


def stop_budget(
    gap: FloatArray,
    v_leader: FloatArray,
    b_leader: FloatArray,
    margin: float | FloatArray = C.SAFETY_MARGIN,
) -> FloatArray:
    """G.2 stopping budget ``C = g + v_L^2 / (2 b_L) - s_m`` (obstacles: v_L = 0, s_m = 0).

    The caller takes the minimum over the leader and every active obstacle.
    """
    with np.errstate(invalid="ignore"):
        budget: FloatArray = gap + v_leader**2 / (2 * b_leader) - margin
    return np.where(np.isinf(gap), np.inf, budget)


def safe_speed(v: FloatArray, budget: FloatArray, b_hat: FloatArray, dt: float) -> FloatArray:
    """G.2 positive root of ``(v + v')/2 dt + v'^2 / (2 b_hat) <= C``.

    ``r = b_hat(-dt/2 + sqrt(dt^2/4 + (2/b_hat)(C - v dt/2)))``; ``-inf`` where the
    radicand is negative (the in-step stop branch of :func:`ballistic` handles it) and
    ``inf`` where ``C`` is ``inf`` (nothing ahead).
    """
    radicand = dt * dt / 4 + (2 / b_hat) * (budget - v * dt / 2)
    with np.errstate(invalid="ignore"):
        root = b_hat * (-dt / 2 + np.sqrt(radicand))
    return np.where(radicand < 0, -np.inf, root)


def ballistic(
    v: FloatArray,
    a: FloatArray,
    budget: FloatArray,
    v_safe: FloatArray,
    b_emerg: FloatArray,
    dt: float,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """G.3 ballistic update (Treiber & Kanagaraj 2015): ``(v', dx, safety_cap_violation)``.

    ``v_safe`` is :func:`safe_speed` (the raw root), ``budget`` the ``C`` it was computed
    from and ``b_emerg`` the follower's ``b_F``. With ``v* = min(v + a dt, max(0, r))``:

    * ``v* > 0``: ``v' = max(v*, v - b_F dt)``, ``dx = (v + v')/2 dt``; a binding floor is
      a violation.
    * otherwise the vehicle stops inside the step: ``dx = v^2 / (2 b*)`` with
      ``b* = min(b_F, max(-a, v/dt, v^2 / (2 max(C, eps)), eps))``; a violation when
      ``v^2 / (2 max(C, eps)) > b_F``. ``v = 0`` stays put.

    A vehicle faster than ``b_F dt`` cannot stop inside the step, so it takes the first
    branch (floor binding, a violation) even when ``v* <= 0``; that case is reachable only
    after a violation (G.2 invariant broken) and keeps every deceleration within ``b_F``
    (assumption A1). Violation tests allow ``eps`` of float noise, so braking exactly at
    ``b_hat = b_F`` is not counted. The applied acceleration is ``(v' - v)/dt``;
    ``v' >= 0`` always.
    """
    eps = C.BALLISTIC_FLOOR
    v_star = np.minimum(v + a * dt, np.maximum(0.0, v_safe))
    floor = v - b_emerg * dt
    moving = (v_star > 0) | (floor > 0)
    v_move = np.maximum(v_star, floor)
    dx_move = (v + v_move) / 2 * dt
    need = v * v / (2 * np.maximum(budget, eps))
    b_star = np.minimum(b_emerg, np.maximum(np.maximum(-a, v / dt), np.maximum(need, eps)))
    dx_stop = v * v / (2 * b_star)
    v_new = np.where(moving, v_move, 0.0)
    dx = np.where(moving, dx_move, dx_stop)
    violation = np.where(moving, floor > v_star + eps, (v > 0) & (need > b_emerg + eps))
    return v_new, dx, violation


def desired_speed(
    net: CompiledNetwork,
    veh: VehicleTable,
    types: VehicleTypes,
    idx: IntArray,
    links: IntArray | None = None,
) -> FloatArray:
    """AG.1 R7 desired speed ``min(type.max_speed, f * link_speed_limit)`` of handles ``idx``.

    On their current links, or on ``links`` (e.g. the lane a vehicle is about to enter).
    The single source used by the engine (``v0`` on link entry) and by delay metrics.
    """
    limit = net.link_speed_limit[veh.link[idx] if links is None else links]
    factor = veh.speed_factor[idx].astype(np.float64)
    return np.minimum(types.max_speed[veh.type_idx[idx]], factor * limit)
