"""Layer 2: compile a resolved scenario into immutable indexed arrays (plan E.3, E.5).

``build_road_graph`` lives in :mod:`urbanflow.network.graph` (imported on demand, since
networkx is slow to import).
"""

from urbanflow.network.compiled import CompiledNetwork, CompileReport
from urbanflow.network.compiler import compile_network
from urbanflow.network.conflicts import ConflictKind, ConflictTable

__all__ = ["CompileReport", "CompiledNetwork", "ConflictKind", "ConflictTable", "compile_network"]
