"""Runtime invariants (plan F.7).

:func:`check_always` runs after every step (I1 finite state, I2 non-negative speed).
:func:`check_debug` runs when ``config.debug_checks`` is set:

* I3 on-link: ``-1e-6 <= pos <= link_length + 1e-6``;
* I4 no longitudinal overlap: every leader gap from the step-4 leader search, recomputed
  on the post-advance state, is ``>= -1e-6``;
* I5 zone exclusivity: no two vehicles on different connectors have bodies (``[pos - len,
  pos]`` on a connector, or on the lane after it mapped back to their locked connector,
  ``lock_conn``) inside the same crossing or merging zone (open intervals shrunk by
  ``1e-6``: a vehicle held at a zone is capped exactly at its ``z_in``), unless one of
  them is ``forced`` (such overlaps are counted in ``zone_conflicts``). Diverging zones are
  not checked;
* I6 conservation: ``generated = backlog + active + arrived + removed``, the table's
  statuses agree, and every running vehicle is on a valid link;
* I7 route consistency: a lane's road is ``route[cursor]``; a connector's movement goes
  from ``route[cursor]`` to ``route[cursor + 1]``;
* I8 no ungated crossing: every vehicle that crossed a stop line this step, and every
  vehicle on a connector, is committed;
* I9 held vehicles stay behind their obstacle: every vehicle held in the step (at a stop
  line or lane end: the lane end; at a zone: its ``z_in``) has its front, mapped onto the
  obstacle's link, at most ``1e-6`` past it, except the exempt ones of :class:`Holds`;
* I10 signal state machine: signalised movement states are r, y, g or G, and a movement
  that was G or g at the end of the previous step is not r now unless it went through
  yellow, i.e. unless the change was a forced ``set_phase`` or its program has yellow = 0;
* I11 identity: uids unique among live vehicles; handle, id and uid maps consistent.

A failure logs at ERROR and raises :class:`InvariantViolation` ``(rule, step, uids,
details)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from urbanflow.core.constants import LOG_MAX_IDS, POSITION_EPS
from urbanflow.core.errors import InvariantViolation
from urbanflow.core.types import (
    BoolArray,
    FloatArray,
    IntArray,
    IntersectionKind,
    SignalState,
    VehicleStatus,
)
from urbanflow.engine.leaders import compute_leaders, expand_csr, remaining_roads
from urbanflow.vehicles.table import VehicleTable

if TYPE_CHECKING:
    from urbanflow.engine.advance import Advance
    from urbanflow.engine.engine import Engine

__all__ = ["Holds", "check_always", "check_debug", "check_signals"]

_log = logging.getLogger("urbanflow.engine")
_WAITING = VehicleStatus.waiting_insert.code
_RUNNING = VehicleStatus.running.code


@dataclass(frozen=True, slots=True)
class Holds:
    """The virtual obstacles of one step's held vehicles (I9)."""

    handles: IntArray
    link: IntArray
    """Link of the obstacle point: the lane (stop line or lane end) or the connector (zone)."""
    pos: FloatArray
    """Obstacle point on ``link``: the lane length, or the zone's ``z_in``."""
    exempt: BoolArray
    """The obstacle appeared this step (not held in the previous one) and ``v^2/(2d) >
    b_emerg``: it cannot be honoured (the overrun counts in ``safety_cap_violations``)."""


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
    """I3-I11 on the committed state of ``engine``.

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
    sizes = engine.routes.lengths
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

    # I9 held vehicles stay behind their obstacle
    if engine.holds is not None:
        bad_h = _past_obstacle(engine, engine.holds)
        if bad_h.size:
            raise _violation("I9", step, veh.uid[bad_h], "held vehicle past its obstacle")

    # I5 zone exclusivity
    pairs = _zone_overlaps(engine, run)
    if pairs:
        uids = sorted({int(veh.uid[h]) for pair in pairs for h in pair})
        raise _violation("I5", step, uids, f"{len(pairs)} pair(s) share a conflict zone")

    # I4 no longitudinal overlap (the step-4 search on the post-advance state)
    remaining = remaining_roads(veh, engine.routes, run)
    leaders = compute_leaders(net, veh, run, engine.types, remaining, engine.sibling_groups)
    bad = (leaders.leader >= 0) & (leaders.gap < -POSITION_EPS)
    if bad.any():
        worst = float(leaders.gap[bad].min())
        raise _violation("I4", step, uid[bad], f"longitudinal overlap (gap {worst:.3f} m)")

    # I10 signal state machine
    check_signals(engine)

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


def _past_obstacle(engine: Engine, holds: Holds) -> IntArray:
    """I9: running, non-exempt held vehicles whose front is past their obstacle point."""
    net, veh = engine.network, engine.vehicles
    n_lanes = net.n_lanes
    keep = veh.active[holds.handles] & ~holds.exempt
    h, at, point = holds.handles[keep], holds.link[keep].astype(np.intp), holds.pos[keep]
    link = veh.link[h].astype(np.intp)
    zone = at >= n_lanes
    ci = np.where(zone, at - n_lanes, 0)
    left = np.where(link >= n_lanes, net.conn_from_lane[np.maximum(link - n_lanes, 0)], -1)
    shift = np.select(
        [
            link == at,
            zone & (link == net.conn_from_lane[ci]),
            zone & (link == net.conn_to_lane[ci]),
            ~zone & (left == at),  # crossed the stop line it was held at
        ],
        [0.0, -net.link_length[net.conn_from_lane[ci]], net.link_length[at], net.link_length[at]],
        np.inf,
    )
    past: IntArray = h[veh.pos[h] + shift > point + POSITION_EPS]
    return past


def _zone_overlaps(engine: Engine, run: IntArray) -> list[tuple[int, int]]:
    """I5: handle pairs on different connectors inside one crossing or merging zone,
    neither of them ``forced``."""
    net, veh, zones = engine.network, engine.vehicles, engine.junctions.zones
    n_lanes = net.n_lanes
    link = veh.link[run].astype(np.intp)
    conn = np.where(link >= n_lanes, link, veh.lock_conn[run])
    k = np.flatnonzero(conn >= 0)
    c, lk = conn[k].astype(np.intp), link[k]
    ci = c - n_lanes
    shift = np.select(
        [lk == c, lk == net.conn_to_lane[ci], lk == net.conn_from_lane[ci]],
        [0.0, net.link_length[c], -net.link_length[net.conn_from_lane[ci]]],
        np.inf,
    )
    front = veh.pos[run[k]] + shift
    rear = front - veh.length[run[k]]
    which, e = expand_csr(zones.ptr, ci)
    # POSITION_EPS: a vehicle held at a zone is capped exactly at its z_in (round-off)
    inside = (rear[which] < zones.zone[e, 1] - POSITION_EPS) & (
        front[which] > zones.zone[e, 0] + POSITION_EPS
    )
    conf, on, who = zones.conflict[e][inside], c[which][inside], run[k[which]][inside]
    pairs: list[tuple[int, int]] = []
    if not conf.size:
        return pairs
    order = np.lexsort((on, conf))
    conf, on, who = conf[order], on[order], who[order]
    starts = np.flatnonzero(np.r_[True, conf[1:] != conf[:-1]])
    for a, b in zip(starts.tolist(), [*starts[1:].tolist(), conf.size], strict=True):
        if on[a] == on[b - 1]:
            continue  # one connector only (car-following separates them)
        first = on[a:b] == on[a]
        pairs += [
            (x, y)
            for x in who[a:b][first].tolist()
            for y in who[a:b][~first].tolist()
            if not (veh.forced[x] or veh.forced[y])
        ]
    return pairs


def check_signals(engine: Engine) -> None:
    """I10 on ``engine.signals`` against the states at the end of the previous step."""
    net, sig = engine.network, engine.signals
    j = net.mov_intersection
    signalised = net.int_kind[j] == IntersectionKind.signalized.code
    cur, prev = sig.movement_state, sig.last_state
    bad = signalised & (cur > SignalState.G.code)
    if not bad.any():
        was_green = prev >= SignalState.g.code
        bad = signalised & was_green & (cur == SignalState.r.code)
        bad &= ~(sig.forced[j] | sig.yellow_zero[j])
        problem = "went from green to red without yellow"
    else:
        problem = "has a state other than r, y, g or G"
    if bad.any():
        ids = ", ".join(net.mov_ids[m] for m in np.flatnonzero(bad)[:LOG_MAX_IDS].tolist())
        raise _violation("I10", engine.step_count, [], f"movement {ids} {problem}")
