"""Validate spatial weighting by enumerating paths, independently of Brandes."""

import math
import random
import unittest
import warnings
import contextlib
import importlib.util
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

from igraph import Graph, InternalError


def enumerated_reference(graph, coords, box, direction):
    """Small simple graphs only: sum unordered pairs and all shortest paths."""
    result = [[0.0] * graph.ecount() for _ in range(3)]
    for source in range(graph.vcount()):
        for target in range(source + 1, graph.vcount()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                paths = graph.get_all_shortest_paths(source, to=target)
            if not paths:
                continue
            dr = [coords[target][axis] - coords[source][axis] for axis in range(3)]
            dr = [d - length * round(d / length) for d, length in zip(dr, box)]
            distance = math.sqrt(sum(d * d for d in dr))
            weights = (1.0, abs(dr[direction]) / distance if distance else 0.0, distance / box[direction])
            for path in paths:
                for v, w in zip(path, path[1:]):
                    edge = graph.get_eid(v, w)
                    for channel, weight in zip(result, weights):
                        channel[edge] += weight / len(paths)
    return result


class SpatialBetweennessTests(unittest.TestCase):
    def assertChannelsClose(self, actual, expected):
        for channel, reference in zip(actual, expected):
            self.assertEqual(len(channel), len(reference))
            for value, ref in zip(channel, reference):
                self.assertTrue(math.isclose(value, ref, rel_tol=1e-12, abs_tol=1e-12), (value, ref))

    def test_enumerated_paths(self):
        rng = random.Random(420)
        graphs = [Graph(), Graph(1), Graph(5), Graph.Ring(4), Graph.Full(5), Graph.Tree(9, 2)]
        for n in range(2, 13):
            edges = [(v, w) for v in range(n) for w in range(v + 1, n) if rng.random() < 0.3]
            rng.shuffle(edges)
            graphs.append(Graph(n, edges))
        box = [10.0, 7.0, 4.0]
        for graph in graphs:
            coords = [[rng.uniform(-20, 20) for _ in range(3)] for _ in range(graph.vcount())]
            for direction in range(3):
                with self.subTest(n=graph.vcount(), m=graph.ecount(), direction=direction):
                    actual = graph.edge_betweenness_spatial(coords, box, direction)
                    self.assertEqual(actual[0], graph.edge_betweenness(directed=False))
                    self.assertChannelsClose(actual, enumerated_reference(graph, coords, box, direction))

    def test_periodic_pair_and_loading_axis(self):
        graph = Graph(2, [(1, 0)])
        coords = [[9, 0, 0], [1, 3, 0]]
        box = [10, 8, 6]
        distance = math.sqrt(13)
        self.assertChannelsClose(graph.edge_betweenness_spatial(coords, box), ([1], [2 / distance], [distance / 10]))
        self.assertChannelsClose(graph.edge_betweenness_spatial(coords, box, 1), ([1], [3 / distance], [distance / 8]))
        self.assertChannelsClose(graph.edge_betweenness_spatial(coords, box, 2), ([1], [0], [distance / 6]))
        shifted = [[29, -16, 6], [-9, 27, -12]]
        self.assertChannelsClose(graph.edge_betweenness_spatial(shifted, box), graph.edge_betweenness_spatial(coords, box))
        self.assertChannelsClose(graph.edge_betweenness_spatial([[0, 0, 0], [5, 4, 3]], box), enumerated_reference(graph, [[0, 0, 0], [5, 4, 3]], box, 0))

    def test_coincident_vertices_loops_and_parallel_edges(self):
        graph = Graph(4, [(2, 1), (0, 1), (0, 1), (1, 1)])
        actual = graph.edge_betweenness_spatial([[0, 0, 0]] * 4, [10, 8, 6])
        self.assertEqual(actual[0], graph.edge_betweenness())
        self.assertEqual(actual[1:], ([0.0] * 4, [0.0] * 4))
        coords = [[0, 0, 0], [1, 0, 0], [3, 0, 0], [0, 0, 0]]
        actual = graph.edge_betweenness_spatial(coords, [10, 8, 6])
        self.assertChannelsClose(actual, ([2, 1, 1, 0], [2, 1, 1, 0], [0.5, 0.2, 0.2, 0]))

    def test_edge_order(self):
        edges = [(3, 1), (2, 0), (1, 0)]
        graph = Graph(4, edges)
        self.assertEqual(graph.get_edgelist(), [tuple(sorted(edge)) for edge in edges])
        coords = [[0, 0, 0], [1, 2, 0], [2, 1, 0], [3, 0, 1]]
        self.assertChannelsClose(graph.edge_betweenness_spatial(coords, [10, 8, 6]), enumerated_reference(graph, coords, [10, 8, 6], 0))

    def test_invalid_inputs(self):
        graph = Graph(2, [(0, 1)])
        coords = [[0, 0, 0], [1, 0, 0]]
        invalid = [([], [10, 8, 6], 0), ([[0, 0], [1, 0, 0]], [10, 8, 6], 0),
                   (coords, [10, 8], 0), (coords, [0, 8, 6], 0), (coords, [-1, 8, 6], 0),
                   (coords, [math.nan, 8, 6], 0), (coords, [math.inf, 8, 6], 0),
                   ([[math.nan, 0, 0], [1, 0, 0]], [10, 8, 6], 0),
                   ([[math.inf, 0, 0], [1, 0, 0]], [10, 8, 6], 0),
                   (coords, [10, 8, 6], -1), (coords, [10, 8, 6], 3), (coords, [10, 8, 6], 0.5)]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises((ValueError, TypeError, InternalError)):
                graph.edge_betweenness_spatial(*args)
        with self.assertRaises((ValueError, InternalError)):
            Graph(2, [(0, 1)], directed=True).edge_betweenness_spatial(coords, [10, 8, 6])
        # A failed call must not poison igraph's cleanup stack or the next call.
        self.assertChannelsClose(graph.edge_betweenness_spatial(coords, [10, 8, 6]), ([1], [1], [0.1]))

    def test_numpy_arrays(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is optional")
        graph = Graph.Ring(4)
        coords = np.arange(24, dtype=float).reshape(4, 6)[:, ::2]
        self.assertChannelsClose(graph.edge_betweenness_spatial(coords, np.array([10, 8, 6]), np.int64(1)), enumerated_reference(graph, coords, [10, 8, 6], 1))


class LammpsSpatialRunnerTests(unittest.TestCase):
    data = """LAMMPS test fixture

3 atoms
2 bonds

-1 9 xlo xhi
0 8 ylo yhi
0 6 zlo zhi

Atoms # full

3 1 1 0 3 0 0 0 0 0
1 1 1 0 9 0 0 1 0 0
2 1 1 0 1 0 0 0 0 0

Velocities

1 0 0 0
2 0 0 0
3 0 0 0

Bonds

8 1 3 2
4 1 2 1
"""

    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError:
            raise unittest.SkipTest("NumPy is needed by the analysis runner")
        cls.np = np
        path = Path(__file__).resolve().parents[1] / "scripts" / "spatial_lammps.py"
        spec = importlib.util.spec_from_file_location("spatial_lammps", path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_reader_and_output_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "network.data"
            path.write_text(self.data)
            coords, box, bond_ids, edges = self.runner.read_lammps_full(path)
            self.np.testing.assert_array_equal(coords[:, 0], [9, 1, 3])
            self.np.testing.assert_array_equal(box, [10, 8, 6])
            self.np.testing.assert_array_equal(bond_ids, [8, 4])
            self.np.testing.assert_array_equal(edges, [[2, 1], [1, 0]])
            with patch("sys.argv", ["spatial_lammps.py", str(path), "--validate-stock"]), contextlib.redirect_stdout(io.StringIO()):
                self.runner.main()
            with self.np.load(path.with_suffix(".spatial.npz")) as output:
                self.np.testing.assert_array_equal(output["bond_ids"], bond_ids)
                self.np.testing.assert_array_equal(output["edges"], edges)
                self.np.testing.assert_array_equal(output["raw_GEBC_igraph"], [2, 2])
                self.np.testing.assert_allclose(output["raw_DBC"], [0.6, 0.6])
                self.np.testing.assert_allclose(output["DBC"], [0.2, 0.2])
            with patch("sys.argv", ["spatial_lammps.py", str(path), "--sample-vertices", "1", "--validate-stock"]), contextlib.redirect_stdout(io.StringIO()):
                self.runner.main()
            with self.np.load(path.with_suffix(".sample.spatial.npz")) as output:
                self.assertEqual(output["raw_GEBC_igraph"].size, 0)

    def test_rejects_invalid_topology_or_cell_input(self):
        bad_data = [self.data.replace("3 1 1 0 3", "2 1 1 0 3"),
                    self.data.replace("8 1 3 2", "8 1 4 2"),
                    self.data.replace("8 1 3 2", "4 1 3 2"),
                    self.data.replace("0 6 zlo zhi", "0 6 zlo zhi\n1 0 0 xy xz yz"),
                    self.data.replace("3 atoms", "4 atoms")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "network.data"
            for data in bad_data:
                path.write_text(data)
                with self.subTest(data=data), self.assertRaises(ValueError):
                    self.runner.read_lammps_full(path)


if __name__ == "__main__":
    unittest.main()
