"""Runtime invariants (plan F.7).

:func:`check_always` runs after every step (I1 finite state, I2 non-negative speed).
:func:`check_debug` runs when ``config.debug_checks`` is set:

* I3 on-link: ``-1e-6 <= pos <= link_length + 1e-6``;
* I4 no longitudinal overlap: every leader gap from the step-4 leader search, recomputed
  on the post-advance state, is ``>= -1e-6``;
* I6 conservation: ``generated = backlog + active + arrived + removed``, the table's
  statuses agree, and every running vehicle is on a valid link;
* I7 route consistency: a lane's road is ``route[cursor]``; a connector's movement goes
  from ``route[cursor]`` to ``route[cursor + 1]``;
* I8 no ungated crossing: every vehicle that crossed a stop line this step, and every
  vehicle on a connector, is committed;
* I11 identity: uids unique among live vehicles; handle, id and uid maps consistent.

A failure logs at ERROR and raises :class:`InvariantViolation` ``(rule, step, uids,
details)``. I5, I9 and I10 arrive with zone locks and signals.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from urbanflow.core.constants import POSITION_EPS
from urbanflow.core.errors import InvariantViolation
from urbanflow.core.types import IntArray, VehicleStatus
from urbanflow.engine.leaders import compute_leaders, remaining_roads
from urbanflow.vehicles.table import VehicleTable

if TYPE_CHECKING:
    from urbanflow.engine.advance import Advance
    from urbanflow.engine.engine import Engine

__all__ = ["check_always", "check_debug"]

_log = logging.getLogger("urbanflow.engine")
_WAITING = VehicleStatus.waiting_insert.code
_RUNNING = VehicleStatus.running.code


def _violation(
    rule: str, step: int, uids: NDArray[Any] | list[int], details: str
) -> InvariantViolation:
    err = InvariantViolation(rule, step, [int(u) for u in uids], details)
    _log.error("%s", err)
    return err


def check_always(veh: VehicleTable, run: IntArray, step: int) -> None:
    """I1 and I2 over the running handles ``run``."""
    pos, speed = veh.pos[run], veh.speed[run]
    bad = ~(np.isfinite(pos) & np.isfinite(speed))
    if bad.any():
        raise _violation("I1", step, veh.uid[run[bad]], "position or speed is not finite")
    bad = speed < 0
    if bad.any():
        raise _violation("I2", step, veh.uid[run[bad]], "negative speed")


def check_debug(engine: Engine, advance: Advance | None = None) -> None:
    """I3, I4, I6, I7, I8 and I11 on the committed state of ``engine``.

    ``advance`` is the step's :class:`~urbanflow.engine.advance.Advance` (for I8's list of
    stop-line crossings); without it only connector vehicles are checked.
    """
    net, veh, step = engine.network, engine.vehicles, engine.step_count
    run = veh.running()
    uid = veh.uid[run]
    link = veh.link[run].astype(np.intp)

    # I6 conservation (first: the other checks index links)
    backlog = len(engine.queues)
    top = veh.top
    live = np.fromiter(veh.id_to_handle.values(), dtype=np.intp, count=len(veh.id_to_handle))
    waiting = int((veh.status[live] == _WAITING).sum())
    total = backlog + run.size + engine.arrived + engine.removed
    if engine.generated != total or waiting != backlog:
        raise _violation(
            "I6",
            step,
            [],
            f"generated {engine.generated} != backlog {backlog} + active {run.size} + "
            f"arrived {engine.arrived} + removed {engine.removed} (waiting rows: {waiting})",
        )
    bad = (link < 0) | (link >= net.n_links)
    if bad.any():
        raise _violation("I6", step, uid[bad], "running vehicle not on a valid link")

    # I3 on-link
    pos = veh.pos[run]
    bad = (pos < -POSITION_EPS) | (pos > net.link_length[link] + POSITION_EPS)
    if bad.any():
        raise _violation("I3", step, uid[bad], "position outside its link")

    # I7 route consistency
    routes = engine.routes.routes
    sizes = np.fromiter((len(r) for r in routes), dtype=np.intp, count=len(routes))
    flat = np.concatenate([*routes, np.zeros(1, dtype=np.int32)])  # padded: base + 1 is valid
    rid, cursor = veh.route_id[run].astype(np.intp), veh.route_cursor[run].astype(np.intp)
    known = (rid >= 0) & (rid < len(routes))
    rid = np.where(known, rid, 0)
    size = np.where(known, sizes[rid] if sizes.size else 0, 0)
    ok = known & (cursor >= 0) & (cursor < size)
    base = np.where(ok, np.cumsum(sizes)[rid] - size + cursor if sizes.size else 0, 0)
    road = flat[base]
    on_lane = link < net.n_lanes
    ok &= ~on_lane | (net.link_road[link] == road)
    conn = np.flatnonzero(~on_lane)
    mov = net.link_movement[link[conn]]
    ok[conn] &= (
        (cursor[conn] + 1 < size[conn])
        & (net.mov_from_road[mov] == road[conn])
        & (net.mov_to_road[mov] == flat[base[conn] + 1])
    )
    if not ok.all():
        raise _violation("I7", step, uid[~ok], "link does not match the route")

    # I8 no ungated crossing
    bad = ~on_lane & ~veh.committed[run]
    if bad.any():
        raise _violation("I8", step, uid[bad], "vehicle on a connector without a commit")
    if advance is not None and not advance.crossed_committed.all():
        crossers = advance.crossed[~advance.crossed_committed]
        raise _violation("I8", step, veh.uid[crossers], "crossed a stop line uncommitted")

    # I4 no longitudinal overlap (the step-4 search on the post-advance state)
    remaining = remaining_roads(veh, engine.routes, run)
    leaders = compute_leaders(net, veh, run, engine.types, remaining, engine.sibling_groups)
    bad = (leaders.leader >= 0) & (leaders.gap < -POSITION_EPS)
    if bad.any():
        worst = float(leaders.gap[bad].min())
        raise _violation("I4", step, uid[bad], f"longitudinal overlap (gap {worst:.3f} m)")

    # I11 identity
    ids = veh.ids
    broken = [
        int(veh.uid[h])
        for vid, h in veh.id_to_handle.items()
        if ids[h] != vid
        or veh.uid_to_id[int(veh.uid[h])] != vid
        or veh.status[h] not in (_WAITING, _RUNNING)
    ]
    if broken or np.unique(veh.uid[live]).size != live.size:
        raise _violation("I11", step, broken, "handle, id and uid maps are inconsistent")
    active = np.flatnonzero(veh.active[:top])
    if not np.array_equal(active, np.sort(live[veh.status[live] == _RUNNING])):
        raise _violation("I11", step, [], "active flags disagree with the running vehicles")
