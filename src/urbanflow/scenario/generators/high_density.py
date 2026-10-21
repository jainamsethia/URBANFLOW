"""``high_density``: a signalised grid loaded close to saturation (plan E.10).

Every entry road gets ``entry_rate = saturation * s * lanes * g / C``, the share
``saturation`` of what its green can discharge: saturation flow ``s`` (1800 veh/h/lane),
green ``g`` per phase and cycle ``C = phases * (g + yellow + all_red)`` (2 phases for the
``two_phase``/``auto`` templates, 4 otherwise). The grid is then built by ``grid`` and the
scenario's ``deadlock_timeout`` set, since queues spill back between junctions.
"""

from __future__ import annotations

from pydantic import Field

from urbanflow.core import constants as C
from urbanflow.scenario.generators import register_generator
from urbanflow.scenario.generators.grid import GridParams
from urbanflow.scenario.generators.grid import generate as grid
from urbanflow.scenario.scenario import Scenario

__all__ = ["HighDensityParams", "entry_rate", "generate"]


class HighDensityParams(GridParams):
    """Parameters of :func:`generate` (the grid's, with the entry rate derived)."""

    entry_rate: float | None = Field(default=None, gt=0)  # type: ignore[assignment]
    """Demand per entry road, veh/h; derived from ``saturation`` when null."""
    saturation: float = Field(default=C.HIGH_DENSITY_SATURATION, gt=0, le=2)
    """Demand / capacity of every entry (v/c)."""
    deadlock_timeout: float = Field(default=C.DEADLOCK_TIMEOUT, ge=0)
    """Watchdog: teleport vehicles stuck this long, s (0 = off)."""


def entry_rate(p: HighDensityParams) -> float:
    """``saturation * s * lanes * g / C`` (see the module docstring)."""
    phases = 2 if p.template in ("two_phase", "auto") else 4
    cycle = phases * (p.green + C.SIGNAL_YELLOW + C.SIGNAL_ALL_RED)
    return p.saturation * C.WEBSTER_SATURATION_FLOW * p.lanes * p.green / cycle


@register_generator("high_density", params=HighDensityParams)
def generate(p: HighDensityParams) -> Scenario:
    """Signalised grid with demand close to capacity."""
    rate = p.entry_rate if p.entry_rate is not None else entry_rate(p)
    fields = {f: getattr(p, f) for f in GridParams.model_fields if f != "entry_rate"}
    base = grid(GridParams(**fields, entry_rate=rate))
    spec = base.spec
    meta = spec.meta.model_copy(
        update={
            "name": f"high_density_{p.rows}x{p.cols}",
            "description": (
                f"{p.rows}x{p.cols} signalized grid at v/c {p.saturation:g}: "
                f"{rate:.0f} veh/h per entry road."
            ),
        }
    )
    simulation = spec.simulation.model_copy(update={"deadlock_timeout": p.deadlock_timeout})
    return Scenario.from_spec(spec.model_copy(update={"meta": meta, "simulation": simulation}))
