"""Resolved vehicle types as per-type parameter arrays (plan E.7 VehicleType, G.2, G.6, G.7).

Type index ``i`` is the position in the resolved scenario's ``vehicle_types`` (built-ins
merged, sorted by id), which is also ``CompiledNetwork.vehicle_types``. Kernels gather
per-vehicle values with ``array[type_idx]``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError

from urbanflow.core import constants as C
from urbanflow.core.errors import ConfigError, NotFoundError, ValidationIssue, suggest
from urbanflow.core.types import FloatArray, IntArray
from urbanflow.scenario.schema import VehicleTypeSpec
from urbanflow.scenario.validate import Loc, issue, issues_from_pydantic

__all__ = ["ParamArrays", "VehicleTypes", "model_param_issues"]


def model_param_issues(
    model_params: Mapping[str, float],
    params_model: type[BaseModel] | None,
    model: str,
    path: Loc,
) -> tuple[list[ValidationIssue], dict[str, float]]:
    """Validate one vehicle type's ``model_params`` against the model's ``Params``.

    Returns ``(issues, values)``: E903 for unknown names (every name is unknown without a
    ``params_model``), else the ``Params`` errors (E004...) at ``path``; ``values`` are
    the validated fields with defaults filled in (empty when there are issues).
    ``VehicleTypes.from_specs`` and the deep check (``urbanflow.check``) both use it.
    """
    known = sorted(params_model.model_fields) if params_model is not None else []
    unknown = [p for p in model_params if p not in known]
    if unknown:
        listed = ", ".join(known) or "none"
        return [issue("E903", (*path, p), p=p, m=model, list=listed) for p in unknown], {}
    if params_model is None:
        return [], {}
    try:
        validated = params_model.model_validate(dict(model_params))
    except ValidationError as exc:
        return issues_from_pydantic(exc, prefix=path, root=params_model), {}
    return [], {k: float(v) for k, v in validated.model_dump().items()}


class ParamArrays:
    """Named float arrays with attribute access: ``p.a``, ``p.b``, ``p.T``, ``p.s0``, ...

    Always present: ``a`` (accel), ``b`` (decel), ``T`` (headway), ``s0`` (min_gap),
    ``b_emerg``, ``b_hat`` (G.2 planning deceleration) and ``delta`` (IDM default unless the
    model defines it), plus every field of the car-following model's ``Params``. Rows are
    vehicle types in :attr:`VehicleTypes.params`; :meth:`gather` gives per-vehicle rows.
    """

    __slots__ = ("_arrays",)

    def __init__(self, arrays: Mapping[str, FloatArray]) -> None:
        self._arrays = dict(arrays)

    def __getattr__(self, name: str) -> FloatArray:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._arrays[name]
        except KeyError:
            raise AttributeError(
                f"no parameter {name!r}{suggest(name, self._arrays)} "
                f"(available: {', '.join(self._arrays)})"
            ) from None

    def __repr__(self) -> str:
        return f"ParamArrays({', '.join(self._arrays)}; n={len(self)})"

    def __len__(self) -> int:
        return len(next(iter(self._arrays.values())))

    def names(self) -> tuple[str, ...]:
        return tuple(self._arrays)

    def gather(self, type_idx: IntArray) -> ParamArrays:
        """Rows ``type_idx`` of every array (per-vehicle parameters from per-type ones)."""
        return ParamArrays({k: v[type_idx] for k, v in self._arrays.items()})


def _f64(values: Sequence[float]) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False, kw_only=True)
class VehicleTypes:
    """Per-type parameter arrays of the resolved vehicle types (index = type index)."""

    specs: tuple[VehicleTypeSpec, ...]
    ids: tuple[str, ...]
    index: Mapping[str, int]
    vclass: NDArray[np.uint8]
    """``VehicleClass`` codes."""
    length: FloatArray
    width: FloatArray
    max_speed: FloatArray
    accel: FloatArray
    decel: FloatArray
    emergency_decel: FloatArray
    min_gap: FloatArray
    headway: FloatArray
    politeness: FloatArray
    lc_threshold: FloatArray
    lc_safe_decel: FloatArray
    b_hat: FloatArray
    """G.2 planning deceleration ``min(own emergency_decel, min over types)``."""
    params: ParamArrays
    """Car-following parameters by type (:class:`ParamArrays`)."""

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        return f"VehicleTypes({', '.join(self.ids)})"

    def type_index(self, type_id: str) -> int:
        """Index of ``type_id``; ``NotFoundError`` with a hint if unknown."""
        try:
            return self.index[type_id]
        except KeyError:
            raise NotFoundError(
                f'unknown vehicle type "{type_id}"{suggest(type_id, self.index)} '
                f"(available: {', '.join(self.ids)})"
            ) from None

    @classmethod
    def from_specs(
        cls,
        specs: Sequence[VehicleTypeSpec],
        params_model: type[BaseModel] | None = None,
        *,
        model: str = "idm",
    ) -> VehicleTypes:
        """Build from the *resolved* types (``scenario.resolved.vehicle_types``).

        ``params_model`` is the car-following model's ``Params``: each type's
        ``model_params`` is validated against it (unknown names raise ``ConfigError`` E903,
        bad values E004...), and its fields, defaults filled in, become :attr:`params`
        columns. ``model`` names the model in messages.
        """
        if not specs:
            raise ConfigError("at least one vehicle type is required")
        issues: list[ValidationIssue] = []
        extras: list[dict[str, float]] = []
        known = sorted(params_model.model_fields) if params_model is not None else []
        for vt in specs:
            base = ("vehicle_types", vt.id, "model_params")
            found, values = model_param_issues(vt.model_params, params_model, model, base)
            issues += found
            extras.append(values)
        if issues:
            raise ConfigError("invalid vehicle type parameters", issues)

        def col(name: str) -> FloatArray:
            return _f64([float(getattr(vt, name)) for vt in specs])

        b_emerg = col("emergency_decel")
        b_hat = np.minimum(b_emerg, b_emerg.min())  # G.2: b_hat <= b_L for every leader
        b_hat.setflags(write=False)
        columns: dict[str, FloatArray] = {
            "a": col("accel"),
            "b": col("decel"),
            "T": col("headway"),
            "s0": col("min_gap"),
            "b_emerg": b_emerg,
            "b_hat": b_hat,
            "delta": _f64([C.IDM_DELTA] * len(specs)),
        }
        for name in known:
            columns[name] = _f64([e[name] for e in extras])
        vclass: NDArray[Any] = np.array([vt.vclass.code for vt in specs], dtype=np.uint8)
        vclass.setflags(write=False)
        ids = tuple(vt.id for vt in specs)
        return cls(
            specs=tuple(specs),
            ids=ids,
            index={t: i for i, t in enumerate(ids)},
            vclass=vclass,
            length=col("length"),
            width=col("width"),
            max_speed=col("max_speed"),
            accel=columns["a"],
            decel=columns["b"],
            emergency_decel=b_emerg,
            min_gap=columns["s0"],
            headway=columns["T"],
            politeness=col("politeness"),
            lc_threshold=col("lc_threshold"),
            lc_safe_decel=col("lc_safe_decel"),
            b_hat=b_hat,
            params=ParamArrays(columns),
        )
