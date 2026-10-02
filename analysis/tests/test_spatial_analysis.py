"""Scientific and end-to-end tests for the reusable analysis package."""

from __future__ import annotations

import json
from pathlib import Path

import igraph as ig
import numpy as np
import pytest

from spatial_betweenness import (
    AnalysisConfig,
    CentralityValidationError,
    Direction,
    LammpsDataReader,
    LammpsFormatError,
    PolymerNetwork,
    SpatialBetweennessAnalysis,
    ValidationReport,
)
from spatial_betweenness.cli import main

DATA = """LAMMPS full-style test network

4 atoms
2 bonds

0 10 xlo xhi
-4 4 ylo yhi
0 6 zlo zhi

Masses

1 1

Atoms # full

30 1 1 0 3 0 0 0 0 0
10 1 1 0 9 0 0 1 0 0
40 1 1 0 7 0 0 0 0 0
20 1 1 0 1 0 0 0 0 0

Velocities

10 0 0 0
20 0 0 0
30 0 0 0
40 0 0 0

Bonds

8 2 30 20
4 1 20 10
"""


@pytest.fixture
def lammps_file(tmp_path: Path) -> Path:
    path = tmp_path / "network.data"
    path.write_text(DATA)
    return path


def test_reader_maps_ids_and_preserves_bond_rows(lammps_file):
    network = LammpsDataReader().read(lammps_file)
    np.testing.assert_array_equal(network.atom_ids, [10, 20, 30, 40])
    np.testing.assert_array_equal(network.coordinates[:, 0], [9, 1, 3, 7])
    np.testing.assert_array_equal(network.bond_ids, [8, 4])
    np.testing.assert_array_equal(network.bond_types, [2, 1])
    np.testing.assert_array_equal(network.edges, [[2, 1], [1, 0]])
    np.testing.assert_array_equal(network.bond_atom_ids, [[30, 20], [20, 10]])
    np.testing.assert_array_equal(network.box_lengths, [10, 8, 6])
    with pytest.raises(ValueError):
        network.coordinates[0, 0] = 123


@pytest.mark.parametrize("before,after", [
    ("30 1 1 0 3", "20 1 1 0 3"),
    ("8 2 30 20", "8 2 99 20"),
    ("8 2 30 20", "4 2 30 20"),
    ("8 2 30 20", "8 0 30 20"),
    ("4 atoms", "5 atoms"),
    ("2 bonds", "1 bonds"),
    ("0 6 zlo zhi", "0 6 zlo zhi\n1 0 0 xy xz yz"),
    ("0 10 xlo xhi", "0 0 xlo xhi"),
    ("Atoms # full", "Atoms # atomic"),
    ("30 1 1 0 3 0 0", "30 1 1 0 nan 0 0"),
    ("4 atoms", "-1 atoms"),
    ("4 atoms", "4 atoms\n4 atoms"),
    ("0 10 xlo xhi", "0 10 xlo xhi\n0 10 xlo xhi"),
])
def test_reader_rejects_malformed_data(lammps_file, before, after):
    lammps_file.write_text(DATA.replace(before, after))
    with pytest.raises(LammpsFormatError):
        LammpsDataReader().read(lammps_file)


@pytest.mark.parametrize("direction,expected_obc,expected_dbc", [
    ("x", [2, 2], [6, 6]),
    ("y", [0, 0], [6, 6]),
    ("z", [0, 0], [6, 6]),
])
def test_known_periodic_path_and_isolated_node(lammps_file, direction, expected_obc, expected_dbc):
    analysis = SpatialBetweennessAnalysis.from_lammps(lammps_file, AnalysisConfig(direction, validate_stock=True))
    result = analysis.run()
    np.testing.assert_array_equal(result.raw_gebc, [2, 2])
    np.testing.assert_allclose(result.raw_obc, expected_obc)
    np.testing.assert_allclose(result.raw_dbc, expected_dbc)
    assert result.neff == 3  # The fourth atom is isolated and excluded from normalization.
    assert result.normalization_factor == 1 / 3
    np.testing.assert_allclose(result.obc, np.asarray(expected_obc) / 3)
    np.testing.assert_allclose(result.dbc, np.asarray(expected_dbc) / 3)
    assert result.validation.allclose
    assert result.validation.max_absolute_difference == 0
    assert result.validation.pearson is None  # Constant edge scores have no correlation.
    assert analysis.inspect()["components"] == 2


@pytest.mark.parametrize("dbc_length", [None, 2.5])
def test_multiple_shortest_paths_against_pair_enumeration(dbc_length):
    edges = np.array([[2, 3], [0, 2], [1, 3], [0, 1]])
    coords = np.array([[9, 0, 0], [1, 2, 0], [3, 0, 1], [4, 3, 0]], dtype=float)
    network = PolymerNetwork(np.arange(1, 5), coords, np.array([8, 3, 5, 1]), np.ones(4, dtype=int), edges, np.array([[0, 10], [0, 8], [0, 6]]))
    graph = ig.Graph(n=4, edges=edges)
    for direction in range(3):
        expected = np.zeros((3, 4))
        for source in range(4):
            for target in range(source + 1, 4):
                paths = graph.get_all_shortest_paths(source, target)
                dr = coords[target] - coords[source]
                dr -= network.box_lengths * np.rint(dr / network.box_lengths)
                r = np.linalg.norm(dr)
                weight = np.array([1, abs(dr[direction]) / r, r])
                for path in paths:
                    for v, w in zip(path, path[1:]):
                        expected[:, graph.get_eid(v, w)] += weight / len(paths)
        result = SpatialBetweennessAnalysis(network, AnalysisConfig(direction, True, dbc_length=dbc_length)).run()
        np.testing.assert_allclose([result.raw_gebc, result.raw_obc, result.raw_dbc], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dbc_length", [None, 2.5])
def test_native_called_once_and_no_global_progress_handler(lammps_file, monkeypatch, dbc_length):
    native = ig.Graph.edge_betweenness_spatial
    calls = []

    def wrapped(graph, *args):
        calls.append(args)
        return native(graph, *args)

    def forbidden_handler(*args):
        pytest.fail("Library API should not alter the global progress handler")

    monkeypatch.setattr(ig.Graph, "edge_betweenness_spatial", wrapped)
    monkeypatch.setattr(ig, "set_progress_handler", forbidden_handler)
    SpatialBetweennessAnalysis.from_lammps(lammps_file, AnalysisConfig(dbc_length=dbc_length)).run()
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][1], [10, 8, 6])


def test_missing_native_method_fails_clearly(lammps_file, monkeypatch):
    monkeypatch.setattr(ig.Graph, "edge_betweenness_spatial", None)
    analysis = SpatialBetweennessAnalysis.from_lammps(lammps_file)
    assert analysis.inspect()["nodes"] == 4
    with pytest.raises(RuntimeError, match="modified igraph build"):
        analysis.run()


def test_stock_mismatch_is_fatal(lammps_file, monkeypatch):
    monkeypatch.setattr(ig.Graph, "edge_betweenness", lambda *args, **kwargs: [999, 999])
    with pytest.raises(CentralityValidationError) as error:
        SpatialBetweennessAnalysis.from_lammps(lammps_file, AnalysisConfig(validate_stock=True)).run()
    assert not error.value.report.allclose


def test_nonfinite_native_outputs_are_fatal(lammps_file, monkeypatch):
    monkeypatch.setattr(ig.Graph, "edge_betweenness_spatial", lambda *args: ([2, 2], [np.nan, 2], [0.6, 0.6]))
    with pytest.raises(ArithmeticError, match="raw_obc"):
        SpatialBetweennessAnalysis.from_lammps(lammps_file).run()


@pytest.mark.parametrize("bond", ["8 2 20 20", "8 2 10 20"])
def test_rejects_loops_and_duplicate_pairs(lammps_file, bond):
    lammps_file.write_text(DATA.replace("8 2 30 20", bond))
    with pytest.raises(ValueError, match="self-loops or duplicate"):
        SpatialBetweennessAnalysis.from_lammps(lammps_file)


@pytest.mark.parametrize("dbc_length", [None, 2.5])
def test_empty_and_single_vertex_graphs(tmp_path, dbc_length):
    for count, row in [(0, ""), (1, "Atoms # full\n\n100 1 1 0 1 2 3\n")]:
        path = tmp_path / "empty.data"
        path.write_text(f"LAMMPS test\n\n{count} atoms\n\n0 10 xlo xhi\n0 8 ylo yhi\n0 6 zlo zhi\n\n{row}")
        result = SpatialBetweennessAnalysis.from_lammps(path, AnalysisConfig(validate_stock=True, dbc_length=dbc_length)).run()
        assert result.neff == 0
        assert result.normalization_factor == 0
        assert result.raw_gebc.size == result.obc.size == result.dbc.size == 0
        assert result.validation.allclose


def test_sample_preserves_original_bond_mapping(lammps_file):
    analysis = SpatialBetweennessAnalysis.from_lammps(lammps_file)
    sample = analysis.sample(2)
    np.testing.assert_array_equal(sample.network.atom_ids, [10, 20])
    np.testing.assert_array_equal(sample.network.bond_ids, [4])
    np.testing.assert_array_equal(sample.network.edges, [[1, 0]])
    np.testing.assert_array_equal(sample.network.bond_atom_ids, [[20, 10]])
    assert sample.inspect()["sample"]
    assert sample.inspect()["sample_parent_nodes_edges"] == (4, 2)
    assert sample.run().raw_gebc.tolist() == [1]
    with pytest.raises(ValueError):
        analysis.sample(0)


def test_portable_archive_and_overwrite_guard(lammps_file, tmp_path):
    result = SpatialBetweennessAnalysis.from_lammps(lammps_file, AnalysisConfig(validate_stock=True)).run()
    path, report = result.save(tmp_path / "nested" / "result.npz")
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["raw_GEBC_igraph"], result.raw_gebc)
        np.testing.assert_array_equal(archive["OBC"], result.obc)
        np.testing.assert_array_equal(archive["DBC"], result.dbc)
        assert archive["dbc_length"].item() == 1.0
        np.testing.assert_array_equal(archive["bond_ids"], [8, 4])
        np.testing.assert_array_equal(archive["atom_ids"][archive["edges"]], archive["bond_atom_ids"])
        assert json.loads(str(archive["metadata"])) == json.loads(report.read_text())
    with pytest.raises(FileExistsError):
        result.save(path)
    result.save(path, overwrite=True)
    assert not list(path.parent.glob(".*.npz.*"))


def test_cli_inspect_and_small_analysis(lammps_file, tmp_path, capsys):
    assert main([str(lammps_file), "--inspect"]) == 0
    assert json.loads(capsys.readouterr().out)["nodes"] == 4
    output = tmp_path / "cli.npz"
    assert main([str(lammps_file), "--direction", "y", "--output", str(output), "--validate-stock", "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out)["gebc_validation"]["allclose"]
    assert main([str(lammps_file), "--output", str(output), "--quiet"]) == 1
    assert main([str(lammps_file), "--sample-vertices", "1", "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out)["edges"] == 0


def test_cannot_overwrite_input_even_with_overwrite_flag(tmp_path):
    path = tmp_path / "network.json"
    path.write_text(DATA)
    result = SpatialBetweennessAnalysis.from_lammps(path).run()
    with pytest.raises(ValueError, match="input LAMMPS"):
        result.save(path.with_suffix(".npz"), overwrite=True)
    assert main([str(path), "--output", str(path.with_suffix(".npz")), "--overwrite", "--quiet"]) == 1
    assert path.read_text() == DATA


@pytest.mark.parametrize("value", ["x", "X", "0", 0, np.int64(0), Direction.X])
def test_direction_aliases(value):
    assert AnalysisConfig(value).direction is Direction.X


@pytest.mark.parametrize("value", [-1, 3, 0.0, True, "north"])
def test_invalid_direction(value):
    with pytest.raises(ValueError):
        AnalysisConfig(value)


def test_validation_shape_and_tolerances():
    with pytest.raises(ValueError):
        ValidationReport.compare(np.array([1]), np.array([1, 2]), rtol=1e-12, atol=1e-8)
    with pytest.raises(ValueError):
        AnalysisConfig(atol=-1)


@pytest.mark.parametrize("direction", ["x", "y", "z"])
@pytest.mark.parametrize("length", [1.0, 2.5, 49.39694662])
def test_custom_dbc_length_preserves_geometry_and_other_channels(lammps_file, direction, length):
    default = SpatialBetweennessAnalysis.from_lammps(lammps_file, AnalysisConfig(direction)).run()
    custom = SpatialBetweennessAnalysis.from_lammps(lammps_file, AnalysisConfig(direction, True, dbc_length=length)).run()
    # Periodic pair distances on the three-node path are 2, 2 and 4.
    # Each edge receives one distance-2 and the distance-4 pair, irrespective of axis.
    np.testing.assert_allclose(custom.raw_dbc, [6, 6], rtol=1e-14)
    np.testing.assert_allclose(custom.dbc, [2 / length, 2 / length], rtol=1e-14)
    np.testing.assert_array_equal(custom.raw_gebc, default.raw_gebc)
    np.testing.assert_array_equal(custom.raw_obc, default.raw_obc)
    assert custom.validation.allclose
    np.testing.assert_array_equal(custom.raw_dbc, default.raw_dbc)
    assert custom.metadata["dbc_length"] == length
    assert custom.metadata["dbc_length_source"] == "explicit"
    assert default.metadata["dbc_length_source"] == "none"


@pytest.mark.parametrize("length", [0, -1, np.nan, np.inf, -np.inf, True, "bond", [1.0], 10**1000])
def test_invalid_dbc_lengths(length):
    with pytest.raises(ValueError, match="dbc_length"):
        AnalysisConfig(dbc_length=length)


def test_custom_length_cli_archive_and_inspection(lammps_file, tmp_path, capsys):
    output = tmp_path / "custom.npz"
    assert main([str(lammps_file), "--dbc-length", "2.5", "--inspect"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["dbc_length"] == 2.5
    assert inspection["box_lengths"] == [10, 8, 6]
    assert main([str(lammps_file), "--dbc-length", "2.5", "--output", str(output), "--validate-stock", "--quiet"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dbc_length"] == 2.5
    assert summary["dbc_length_source"] == "explicit"
    with np.load(output, allow_pickle=False) as archive:
        assert archive["dbc_length"].item() == 2.5
        np.testing.assert_allclose(archive["raw_DBC"], [6, 6])
        np.testing.assert_allclose(archive["DBC"], [0.8, 0.8])
        assert archive["raw_dbc_weight"].item() == "distance"
        assert archive["schema_version"].item() == 2
        assert json.loads(str(archive["metadata"]))["dbc_length"] == 2.5
    assert json.loads(output.with_suffix(".json").read_text())["dbc_length"] == 2.5
    assert main([str(lammps_file), "--dbc-length", "0", "--quiet"]) == 1


def test_reuse_same_raw_dbc_for_multiple_lengths(lammps_file):
    result = SpatialBetweennessAnalysis.from_lammps(lammps_file).run()
    for length in (1.0, 2.5, 49.39694662):
        np.testing.assert_allclose(result.dbc_for_length(length), [6 / length, 6 / length])
        np.testing.assert_allclose(result.dbc_for_length(length, normalized=True), [2 / length, 2 / length])
    np.testing.assert_array_equal(result.raw_dbc, [6, 6])
    np.testing.assert_array_equal(result.dbc, [2, 2])
    for length in (0, -1, np.nan, None):
        with pytest.raises(ValueError):
            result.dbc_for_length(length)


def test_rejects_old_native_dbc_convention(lammps_file, monkeypatch):
    monkeypatch.delattr(ig._igraph, "SPATIAL_DBC_WEIGHT")
    with pytest.raises(RuntimeError, match="outdated"):
        SpatialBetweennessAnalysis.from_lammps(lammps_file).run()
