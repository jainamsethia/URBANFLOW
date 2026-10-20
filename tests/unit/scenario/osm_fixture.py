"""A small hand-written OpenStreetMap extract (near 18.52 N, 73.85 E) for importer tests.

Local layout in metres (x east, y north):

* main road, ``highway=primary``, ``lanes=4``, ``maxspeed=50``, y = 0 from x = -400 to 400,
  drawn as three ways: [-400, 0], [0, 300] and [300, 400] (the last two only split at
  x = 300 because of a tag change on nothing relevant: a joint, not a junction);
* cross streets, ``highway=residential``, at x = -200, 0, 200 from y = -300 to y = 300; the
  one at x = -200 meets the main road at two nodes 4 m apart (x = -200 and -196);
  the one at x = 200 has ``maxspeed=30 mph``;
* a one-way street, ``highway=tertiary``, ``oneway=yes``, y = 150 from x = -400 to 400;
* a ``highway=traffic_signals`` node on the main road at x = -5 (the x = 0 crossing);
* ignored: an ``access=private`` road, a footway and a service road.
"""

from __future__ import annotations

import math

LAT0, LON0 = 18.52, 73.85
_KY = 6_371_008.8 * math.pi / 180
_KX = _KY * math.cos(math.radians(LAT0))


class _Osm:
    def __init__(self) -> None:
        self.nodes: dict[tuple[float, float], int] = {}
        self.tags: dict[int, dict[str, str]] = {}
        self.ways: list[tuple[int, list[int], dict[str, str]]] = []

    def node(self, x: float, y: float, **tags: str) -> int:
        key = (x, y)
        if key not in self.nodes:
            self.nodes[key] = 1000 + len(self.nodes)
        nid = self.nodes[key]
        if tags:
            self.tags[nid] = tags
        return nid

    def way(self, points: list[tuple[float, float]], **tags: str) -> None:
        ids = [self.node(x, y) for x, y in points]
        self.ways.append((500 + len(self.ways), ids, tags))

    def xml(self) -> str:
        out = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="test">']
        for (x, y), nid in sorted(self.nodes.items(), key=lambda kv: kv[1]):
            lat, lon = LAT0 + y / _KY, LON0 + x / _KX
            tags = self.tags.get(nid, {})
            body = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items())
            out.append(f'<node id="{nid}" lat="{lat:.8f}" lon="{lon:.8f}">{body}</node>')
        for wid, ids, tags in self.ways:
            nds = "".join(f'<nd ref="{i}"/>' for i in ids)
            body = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items())
            out.append(f'<way id="{wid}">{nds}{body}</way>')
        out.append("</osm>")
        return "\n".join(out)


def town() -> str:
    o = _Osm()
    main = {"highway": "primary", "lanes": "4", "maxspeed": "50", "name": "Main Road"}
    o.node(-5, 0, highway="traffic_signals")
    o.way([(-400, 0), (-200, 0), (-196, 0), (-100, 0), (-5, 0), (0, 0)], **main)
    o.way([(0, 0), (100, 0), (200, 0), (300, 0)], **main)
    o.way([(300, 0), (400, 0)], **main)
    res = {"highway": "residential"}
    o.way([(-200, 300), (-200, 150), (-200, 0)], **res)
    o.way([(-196, 0), (-196, -300)], **res)
    o.way([(0, 300), (0, 150), (0, 0), (0, -300)], **res)
    o.way([(200, 300), (200, 150), (200, 0), (200, -300)], **res, maxspeed="30 mph")
    o.way(
        [(-400, 150), (-200, 150), (0, 150), (200, 150), (400, 150)],
        highway="tertiary",
        oneway="yes",
    )
    o.way([(100, 0), (100, -80)], highway="residential", access="private")
    o.way([(-100, 0), (-100, -60)], highway="footway")
    o.way([(-100, 150), (-100, 220)], highway="service")
    return o.xml()
