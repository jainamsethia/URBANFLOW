"""Leaders and gaps of every running vehicle (plan F.1 step 4).

The same-link leader comes from one stable ``lexsort((uid, pos, link))``. Then, keeping
the minimum gap per vehicle:

* **lookahead** along planned links while the distance to a link's start is below
  ``max(200 m, v^2/(2b) + vT + s0)``, stopping at the first stop line the vehicle is not
  committed to: a committed lane vehicle looks through its planned connector into the
  connector's ``to_lane``; a connector vehicle looks into its ``to_lane``. The stop line at
  the end of that ``to_lane`` (unless it is the route's last road) is reported as
  :attr:`Leaders.stop_gap`, a G.2 cap obstacle;
* **diverge siblings**: for a vehicle on lane ``l`` planning ``c_i``, the last vehicle on
  ``c_i`` and on every other connector ``c`` leaving ``l`` while its rear is within
  ``d_sep(c_i, c)`` (gap ``L_l - pos + pos_j - len_j``); for a vehicle on connector ``c``,
  the nearest vehicle ahead on each diverging sibling while its rear is within ``d_sep``;
* **merge siblings**: a vehicle ``j`` on a merging sibling ``c'`` of connector ``c`` leads
  ``i`` on ``c`` iff ``L_c' - pos_j < L_c - pos_i`` (gap ``(L_c - pos_i) - (L_c' - pos_j) -
  len_j``).

Rears are ``pos - len``. A rear below 0 still hangs over the upstream lane, so it counts as
inside every diverge zone; a vehicle without a planned connector therefore still sees
connector vehicles whose rear blocks its lane end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from urbanflow.core import constants as C
from urbanflow.core.types import BoolArray, FloatArray, IntArray, UIntArray
from urbanflow.network.compiled import CompiledNetwork
from urbanflow.network.conflicts import ConflictKind
from urbanflow.routing.base import RouteTable
from urbanflow.vehicles.table import VehicleTable
from urbanflow.vehicles.types import VehicleTypes

__all__ = [
    "ConnectorConflicts",
    "Leaders",
    "SiblingGroups",
    "compute_leaders",
    "connector_conflicts",
    "expand_csr",
    "remaining_roads",
]

_KEY = C.LC_SORT_KEY_SCALE  # combined sort key link * K + pos (G.4)


@dataclass(frozen=True, slots=True)
class ConnectorConflicts:
    """Conflicts of each connector restricted to some kinds, CSR by connector index.

    Entries of connector ``c`` (index = link id - n_lanes) are ``ptr[c]:ptr[c+1]``, sorted
    by ``z_in`` on ``c``.
    """

    ptr: IntArray
    other: IntArray
    """Link id of the other connector."""
    kind: UIntArray
    zone: FloatArray
    """``[z_in, z_out]`` on the owning connector, shape ``(n, 2)``."""
    other_zone: FloatArray
    """``[z_in, z_out]`` on the other connector."""


def connector_conflicts(
    net: CompiledNetwork, kinds: tuple[ConflictKind, ...]
) -> ConnectorConflicts:
    """The ``conn_conf`` CSR of ``net`` filtered to ``kinds``, with zones on both sides."""
    k = net.conn_conf
    owner = np.repeat(np.arange(net.n_conn), np.diff(net.conn_conf_ptr))
    side_a = net.conn_conf_side == 0
    keep = np.isin(net.conf_kind[k], [int(x) for x in kinds])
    zone_a, zone_b = net.conf_zone_a[k], net.conf_zone_b[k]
    return ConnectorConflicts(
        ptr=np.searchsorted(owner[keep], np.arange(net.n_conn + 1)),
        other=np.where(side_a, net.conf_b[k], net.conf_a[k])[keep],
        kind=net.conf_kind[k][keep],
        zone=np.where(side_a[:, None], zone_a, zone_b)[keep],
        other_zone=np.where(side_a[:, None], zone_b, zone_a)[keep],
    )


@dataclass(frozen=True, slots=True)
class SiblingGroups:
    """Static diverge and merge groups of every connector (built once per network)."""

    diverging: ConnectorConflicts
    merging: ConnectorConflicts

    @classmethod
    def build(cls, net: CompiledNetwork) -> SiblingGroups:
        return cls(
            connector_conflicts(net, (ConflictKind.diverging,)),
            connector_conflicts(net, (ConflictKind.merging,)),
        )


@dataclass(frozen=True, slots=True)
class Leaders:
    """Step-local leader data. Per-vehicle arrays are aligned with the ``run`` argument."""

    order: IntArray
    """Running handles sorted by ``(link, pos, uid)``."""
    perm: IntArray
    """The same order as positions in ``run`` (``order == run[perm]``)."""
    leader: IntArray
    """Leader handle, -1 if none."""
    gap: FloatArray
    """Bumper-to-bumper gap to the leader, m; ``inf`` if none."""
    leader_speed: FloatArray
    """Leader speed, m/s (0 if none)."""
    stop_gap: FloatArray
    """Distance to the stop line that ended the lookahead beyond the current link, m;
    ``inf`` if none (G.2 cap obstacle with ``v_L = 0``, ``s_m = 0``)."""
    tail: IntArray
    """Per link: handle of its most upstream vehicle (smallest ``pos``), -1 if empty."""


def expand_csr(ptr: IntArray, rows: IntArray) -> tuple[IntArray, IntArray]:
    """Every CSR entry of ``rows``: ``(index into rows, entry index)`` pairs, in row order."""
    start = ptr[rows]
    count = ptr[rows + 1] - start
    which = np.repeat(np.arange(rows.size), count)
    offset = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    return which, np.repeat(start, count) + offset


def remaining_roads(veh: VehicleTable, routes: RouteTable, run: IntArray) -> IntArray:
    """Roads after the current one on each vehicle's route (0 on the last road).

    A connector vehicle's current road is the one it came from (``route_cursor`` advances
    on lane entry), so 1 means its ``to_lane`` is on the last road.
    """
    lengths = np.fromiter((len(r) for r in routes.routes), dtype=np.intp, count=len(routes))
    if lengths.size == 0:
        return np.zeros(run.size, dtype=np.intp)
    return lengths[veh.route_id[run]] - 1 - veh.route_cursor[run].astype(np.intp)


def compute_leaders(
    net: CompiledNetwork,
    veh: VehicleTable,
    run: IntArray,
    types: VehicleTypes,
    remaining: IntArray,
    groups: SiblingGroups,
) -> Leaders:
    """F.1 step 4 for the running handles ``run`` (``remaining`` from :func:`remaining_roads`).

    Every candidate leader (same-link successor, lookahead link tails, diverge and merge
    siblings) is collected with its gap; the minimum gap wins, ties by the leader's uid.
    """
    n, n_lanes = run.size, net.n_lanes
    if n == 0:
        none = np.zeros(0, dtype=np.intp)
        empty = np.zeros(0)
        tails = np.full(net.n_links, -1, dtype=np.intp)
        return Leaders(none, none, none, empty, empty, empty, tails)
    link = veh.link[run].astype(np.intp)
    pos = veh.pos[run]
    length = veh.length[run].astype(np.float64)
    speed = veh.speed[run]
    uid = veh.uid[run]
    rear = pos - length
    perm = np.lexsort((uid, pos, link))
    slink = link[perm]
    tail = np.full(net.n_links, -1, dtype=np.intp)
    starts = np.flatnonzero(np.r_[True, slink[1:] != slink[:-1]])
    tail[slink[starts]] = perm[starts]
    key = slink * _KEY + pos[perm]
    lengths = net.link_length
    d_end = lengths[link] - pos
    ti = veh.type_idx[run]
    reach = np.maximum(
        C.LOOKAHEAD_MIN_DISTANCE,
        speed**2 / (2 * types.decel[ti]) + speed * types.headway[ti] + types.min_gap[ti],
    )
    nc = veh.next_conn[run].astype(np.intp)
    on_lane = link < n_lanes
    parts: list[tuple[IntArray, IntArray, FloatArray]] = []

    def add(i: IntArray, j: IntArray, gap: FloatArray, ok: BoolArray | None = None) -> None:
        if ok is not None:
            i, j, gap = i[ok], j[ok], gap[ok]
        parts.append((i, j, gap))

    def first_above(links: IntArray, x: FloatArray) -> IntArray:
        """Position of the first vehicle on ``links`` with ``pos > x``, or -1."""
        raw = np.searchsorted(key, links * _KEY + x, side="right")
        idx = np.minimum(raw, n - 1)
        return np.where((raw < n) & (slink[idx] == links), perm[idx], -1)

    def lookahead(i: IntArray, links: IntArray, offset: FloatArray) -> None:
        j = tail[links]
        ok = (j >= 0) & (offset < reach[i])
        add(i, j, offset + rear[j], ok)

    # same link: the successor in the sorted order
    same = slink[1:] == slink[:-1]
    ahead_i, ahead_j = perm[:-1][same], perm[1:][same]
    add(ahead_i, ahead_j, rear[ahead_j] - pos[ahead_i])

    # lookahead: connector vehicles into their to_lane
    stop_gap = np.full(n, np.inf)
    conn_i = np.flatnonzero(~on_lane)
    conn_c = link[conn_i]
    to = net.conn_to_lane[conn_c - n_lanes].astype(np.intp)
    lookahead(conn_i, to, d_end[conn_i])
    far = remaining[conn_i] >= 2
    stop_gap[conn_i[far]] = d_end[conn_i[far]] + lengths[to[far]]
    # committed lane vehicles through their planned connector into its to_lane
    lane_c = np.flatnonzero(on_lane & veh.committed[run] & (nc >= 0))
    planned = nc[lane_c]
    to2 = net.conn_to_lane[planned - n_lanes].astype(np.intp)
    lookahead(lane_c, planned, d_end[lane_c])
    offset2 = d_end[lane_c] + lengths[planned]
    lookahead(lane_c, to2, offset2)
    far2 = remaining[lane_c] >= 2
    stop_gap[lane_c[far2]] = offset2[far2] + lengths[to2[far2]]

    # diverge siblings of lane vehicles: every connector leaving the lane...
    lane_i = np.flatnonzero(on_lane)
    which, e = expand_csr(net.lane_out_ptr, link[lane_i])
    i = lane_i[which]
    cc = net.lane_out_conn[e]
    j = tail[cc]
    limit = np.where(cc == nc[i], np.inf, 0.0)
    add(i, j, d_end[i] + rear[j], (j >= 0) & (rear[j] <= limit))
    # ...and the diverging siblings of the planned connector within d_sep
    div = groups.diverging
    lane_p = lane_i[nc[lane_i] >= 0]
    which, e = expand_csr(div.ptr, nc[lane_p] - n_lanes)
    i = lane_p[which]
    j = tail[div.other[e]]
    add(i, j, d_end[i] + rear[j], (j >= 0) & (rear[j] <= div.other_zone[e, 1]))
    # diverge siblings of connector vehicles: nearest vehicle ahead within d_sep
    which, e = expand_csr(div.ptr, conn_c - n_lanes)
    i = conn_i[which]
    j = first_above(div.other[e], pos[i])
    add(i, j, rear[j] - pos[i], (j >= 0) & (rear[j] <= div.other_zone[e, 1]))
    # merge siblings of connector vehicles, aligned by distance to the end
    mrg = groups.merging
    which, e = expand_csr(mrg.ptr, conn_c - n_lanes)
    i = conn_i[which]
    other = mrg.other[e]
    rem = d_end[i]
    j = first_above(other, lengths[other] - rem)
    add(i, j, rem - (lengths[other] - pos[j]) - length[j], j >= 0)

    cand_i = np.concatenate([p[0] for p in parts])
    cand_j = np.concatenate([p[1] for p in parts])
    cand_gap = np.concatenate([p[2] for p in parts])
    best = np.lexsort((uid[cand_j], cand_gap, cand_i))
    first = best[np.r_[True, cand_i[best][1:] != cand_i[best][:-1]]] if best.size else best
    lead = np.full(n, -1, dtype=np.intp)
    gap = np.full(n, np.inf)
    lead[cand_i[first]] = cand_j[first]
    gap[cand_i[first]] = cand_gap[first]
    has = lead >= 0
    return Leaders(
        order=run[perm],
        perm=perm,
        leader=np.where(has, run[lead], -1),
        gap=gap,
        leader_speed=np.where(has, speed[lead], 0.0),
        stop_gap=stop_gap,
        tail=np.where(tail >= 0, run[tail], -1),
    )
