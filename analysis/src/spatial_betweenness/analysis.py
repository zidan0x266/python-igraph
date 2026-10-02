"""One native, single-threaded GEBC/OBC/DBC traversal per analysis."""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import igraph as ig
import numpy as np

from .lammps import LammpsDataReader
from .models import AnalysisConfig, PolymerNetwork
from .results import CentralityResult, ValidationReport


class CentralityValidationError(RuntimeError):
    """Custom GEBC disagrees with stock; spatial results must not be accepted."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(f"GEBC validation failed: {report.to_dict()}")


class SpatialBetweennessAnalysis:
    """Prepare a graph once and compute all three centrality channels together.

    The graph is undirected, simple and at bead/bond level. Shortest paths are
    always unweighted; geometry only weights source-target contributions.
    The library does not install global igraph progress handlers.
    """

    def __init__(self, network: PolymerNetwork, config: AnalysisConfig | None = None) -> None:
        self.network = network
        self.config = config or AnalysisConfig()
        self._graph = ig.Graph(n=network.n_atoms, edges=network.edges, directed=False)
        if not self._graph.is_simple():
            raise ValueError("Expected a simple atom/bond graph; found self-loops or duplicate atom-pair bonds")
        edge_ids = np.asarray(self._graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
        if not np.array_equal(edge_ids, np.sort(network.edges, axis=1)):
            raise RuntimeError("igraph edge IDs do not match LAMMPS bond-row order")
        self._neff = int(np.count_nonzero(self._graph.degree()))
        self._sample_parent: tuple[int, int] | None = None

    @classmethod
    def from_lammps(cls, path: str | Path, config: AnalysisConfig | None = None) -> SpatialBetweennessAnalysis:
        return cls(LammpsDataReader().read(path), config)

    @property
    def dbc_length(self) -> float | None:
        """Optional postprocessing denominator; never applied to raw DBC."""
        return self.config.dbc_length

    def sample(self, n_vertices: int) -> SpatialBetweennessAnalysis:
        """Return a BFS-neighborhood induced subgraph for debugging, not an estimate.

        The seed is the smallest atom ID. If its component is smaller than the
        requested size, all vertices reachable from that seed are used.
        """
        if isinstance(n_vertices, bool) or not isinstance(n_vertices, (int, np.integer)) or n_vertices < 1:
            raise ValueError("Sample size must be a positive integer")
        if self.network.n_atoms == 0:
            raise ValueError("Cannot sample an empty graph")
        vertices = np.asarray(self._graph.bfs(0)[0][:n_vertices], dtype=np.int64)
        result = SpatialBetweennessAnalysis(self.network.subset(vertices), self.config)
        result._sample_parent = (self.network.n_atoms, self.network.n_bonds)
        return result

    def inspect(self) -> dict[str, object]:
        """Input/topology metadata; no centrality calculation is performed."""
        return {
            "schema_version": 2,
            "raw_dbc_weight": "distance",
            "input": str(self.network.source) if self.network.source else None,
            "nodes": self.network.n_atoms,
            "edges": self.network.n_bonds,
            "neff": self._neff,
            "components": len(self._graph.connected_components()),
            "box_lengths": self.network.box_lengths.tolist(),
            "direction": int(self.config.direction),
            "loading_axis": self.config.direction.name.lower(),
            "dbc_length": self.dbc_length,
            "dbc_length_source": "none" if self.config.dbc_length is None else "explicit",
            "edge_order_verified": True,
            "graph_level": "atom/bond",
            "pair_convention": "unordered, raw undirected",
            "periodic_axes": ["x", "y", "z"],
            "sample": self._sample_parent is not None,
            "sample_parent_nodes_edges": self._sample_parent,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "python_igraph": ig.__version__,
            "c_igraph": ig.__igraph_version__,
        }

    def run(self) -> CentralityResult:
        """Compute raw channels and optionally check GEBC before returning results."""
        native = getattr(self._graph, "edge_betweenness_spatial", None)
        if not callable(native):
            raise RuntimeError("The modified igraph build is required. Use .spatial-venv/bin/python or run bash python-igraph/scripts/build_spatial.sh first.")
        if getattr(ig._igraph, "SPATIAL_DBC_WEIGHT", None) != "distance":
            raise RuntimeError("The native DBC convention is outdated. Rebuild both libraries with bash python-igraph/scripts/build_spatial.sh before using distance-only raw DBC.")
        metadata = self.inspect()
        metadata["started_at_utc"] = datetime.now(UTC).isoformat()
        start = perf_counter()
        raw_gebc, raw_obc, raw_dbc = [np.asarray(values, dtype=np.float64) for values in native(self.network.coordinates, self.network.box_lengths, int(self.config.direction))]
        metadata["spatial_seconds"] = perf_counter() - start
        validation = None
        if self.config.validate_stock:
            start = perf_counter()
            stock = np.asarray(self._graph.edge_betweenness(directed=False), dtype=np.float64)
            metadata["stock_seconds"] = perf_counter() - start
            validation = ValidationReport.compare(raw_gebc, stock, rtol=self.config.rtol, atol=self.config.atol)
            if not validation.allclose:
                raise CentralityValidationError(validation)
        result = CentralityResult(self.network, raw_gebc, raw_obc, raw_dbc, self._neff, metadata, validation)
        try:
            import resource
        except ImportError:
            metadata["peak_process_rss_mib"] = None
        else:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metadata["peak_process_rss_mib"] = rss / (1024**2 if platform.system() == "Darwin" else 1024)
        return result
