#!/usr/bin/env python3
"""Calculate native spatial betweenness from an orthorhombic LAMMPS full data file."""

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import igraph as ig
import numpy as np


def read_lammps_full(path):
    """Read coordinates by atom ID and bonds in file order, using O(V + E) memory."""
    natoms = nbonds = None
    bounds = {}
    coords = edges = bond_ids = seen = None
    atom_count = bond_count = 0
    section = None
    with open(path) as stream:
        for line in stream:
            data, _, comment = line.partition("#")
            fields = data.split()
            if not fields:
                continue
            if len(fields) == 2 and fields[1] in ("atoms", "bonds"):
                if fields[1] == "atoms":
                    natoms = int(fields[0])
                else:
                    nbonds = int(fields[0])
                continue
            if len(fields) == 4 and fields[2:] in (["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]):
                bounds[fields[2][0]] = (float(fields[0]), float(fields[1]))
                continue
            if fields[-3:] == ["xy", "xz", "yz"] or fields[-1] in ("avec", "bvec", "cvec"):
                raise ValueError("Only orthorhombic boxes are supported")
            if fields[0] == "Atoms":
                if natoms is None or natoms < 0 or coords is not None:
                    raise ValueError("Missing/invalid atom count or repeated Atoms section")
                if comment.strip() and comment.strip() != "full":
                    raise ValueError("Expected Atoms # full: id mol type q x y z [ix iy iz]")
                coords = np.empty((natoms, 3), dtype=np.float64)
                seen = np.zeros(natoms, dtype=bool)
                section = "atoms"
                continue
            if fields[0] == "Bonds":
                if nbonds is None or nbonds < 0 or edges is not None:
                    raise ValueError("Missing/invalid bond count or repeated Bonds section")
                edges = np.empty((nbonds, 2), dtype=np.int64)
                bond_ids = np.empty(nbonds, dtype=np.int64)
                section = "bonds"
                continue
            if fields[0][0].isalpha():
                section = None
                continue
            if section == "atoms":
                if len(fields) not in (7, 10):
                    raise ValueError("Expected 7 or 10 columns in Atoms # full")
                atom = int(fields[0]) - 1
                if atom < 0 or atom >= natoms or seen[atom]:
                    raise ValueError("Atom IDs must be unique and consecutive from 1 through natoms")
                coords[atom] = [float(value) for value in fields[4:7]]
                seen[atom] = True
                atom_count += 1
            elif section == "bonds":
                if len(fields) != 4 or bond_count >= nbonds:
                    raise ValueError("Invalid Bonds section")
                bond_ids[bond_count] = int(fields[0])
                edges[bond_count] = [int(fields[2]) - 1, int(fields[3]) - 1]
                bond_count += 1
    if coords is None or edges is None or atom_count != natoms or bond_count != nbonds:
        raise ValueError("Atoms/Bonds sections do not match declared counts")
    if set(bounds) != {"x", "y", "z"}:
        raise ValueError("Missing orthorhombic box bounds")
    box = np.array([bounds[axis][1] - bounds[axis][0] for axis in "xyz"])
    if not np.isfinite(coords).all() or not np.isfinite(box).all() or np.any(box <= 0):
        raise ValueError("Coordinates must be finite and box lengths finite and positive")
    if edges.size and (edges.min() < 0 or edges.max() >= natoms):
        raise ValueError("Bond endpoint outside the atom-ID range")
    if len(np.unique(bond_ids)) != nbonds:
        raise ValueError("Bond IDs must be unique")
    return coords, box, bond_ids, edges


def comparison(actual, reference):
    difference = np.abs(actual - reference)
    scale = np.maximum(np.abs(reference), 1e-15)
    pearson = None
    if actual.size > 1 and np.std(actual) > 0 and np.std(reference) > 0:
        pearson = float(np.corrcoef(actual, reference)[0, 1])
    return {
        "max_abs_difference": float(difference.max(initial=0)),
        "mean_abs_difference": float(difference.mean()) if difference.size else 0.0,
        "max_relative_difference_floor_1e-15": float((difference / scale).max(initial=0)),
        "pearson": pearson,
        "allclose": bool(np.allclose(actual, reference, rtol=1e-12, atol=1e-8)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--direction", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--output", type=Path, help="Output .npz; default is <input>.spatial.npz")
    parser.add_argument("--inspect", action="store_true", help="Check data, topology and edge order without centrality")
    parser.add_argument("--validate-stock", action="store_true", help="Also run stock GEBC and require agreement")
    parser.add_argument("--sample-vertices", type=int, help="Debug only: use an induced subgraph from a BFS neighborhood")
    args = parser.parse_args()
    coords, box, bond_ids, edges = read_lammps_full(args.data)
    graph = ig.Graph(n=len(coords), edges=edges, directed=False)
    if not graph.is_simple():
        raise ValueError("Expected a simple atom/bond graph; found loops or duplicate bonds")
    if not np.array_equal(np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2), np.sort(edges, axis=1)):
        raise AssertionError("igraph edge IDs do not match original LAMMPS bond order")
    atom_ids = np.arange(1, graph.vcount() + 1, dtype=np.int64)
    if args.sample_vertices is not None:
        if args.sample_vertices < 1 or graph.vcount() == 0:
            raise ValueError("--sample-vertices requires a positive size and nonempty graph")
        order = graph.bfs(0)[0][:args.sample_vertices]
        selected = np.zeros(graph.vcount(), dtype=bool)
        selected[order] = True
        keep_edges = selected[edges].all(axis=1)
        vertices = np.flatnonzero(selected)
        remap = np.full(graph.vcount(), -1, dtype=np.int64)
        remap[vertices] = np.arange(len(vertices))
        original_edges = edges[keep_edges]
        graph = ig.Graph(n=len(vertices), edges=remap[original_edges], directed=False)
        if not np.array_equal(np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2), np.sort(remap[original_edges], axis=1)):
            raise AssertionError("Sample graph edge ordering changed")
        coords, atom_ids = coords[vertices], atom_ids[vertices]
        edges, bond_ids = original_edges, bond_ids[keep_edges]
    neff = int(np.count_nonzero(graph.degree()))
    metadata = {
        "input": str(args.data.resolve()), "python": platform.python_version(),
        "platform": platform.platform(), "python_igraph": ig.__version__, "c_igraph": ig.__igraph_version__,
        "nodes": graph.vcount(), "edges": graph.ecount(), "neff": neff,
        "components": len(graph.connected_components()), "box_lengths": box.tolist(),
        "direction": args.direction, "edge_order_verified": True,
        "sample_vertices_requested": args.sample_vertices,
    }
    print(json.dumps(metadata, indent=2), flush=True)
    if args.inspect:
        return
    if not hasattr(graph, "edge_betweenness_spatial"):
        raise RuntimeError("Use the isolated environment containing the modified igraph build")
    last_report = [None, -10.0]

    def progress(message, percent):
        if message != last_report[0] or percent >= last_report[1] + 5 or percent == 100:
            print(f"{message}{percent:.0f}%", flush=True)
            last_report[:] = [message, percent]

    ig.set_progress_handler(progress)
    try:
        start = time.perf_counter()
        raw_gebc, raw_obc, raw_dbc = [np.asarray(values) for values in graph.edge_betweenness_spatial(coords, box, args.direction)]
        metadata["spatial_seconds"] = time.perf_counter() - start
        if not all(np.isfinite(values).all() for values in (raw_gebc, raw_obc, raw_dbc)):
            raise ArithmeticError("Native calculation returned nonfinite values")
        if args.validate_stock:
            start = time.perf_counter()
            stock = np.asarray(graph.edge_betweenness(directed=False))
            metadata["stock_seconds"] = time.perf_counter() - start
            metadata["gebc_validation"] = comparison(raw_gebc, stock)
            if not metadata["gebc_validation"]["allclose"]:
                raise AssertionError(f"GEBC validation failed: {metadata['gebc_validation']}")
    finally:
        ig.set_progress_handler(None)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata["peak_rss_mib"] = peak_rss / (1024 * 1024 if platform.system() == "Darwin" else 1024)
    # There are no contributing pairs when fewer than two vertices have degree > 0.
    norm_factor = 2.0 / (neff * (neff - 1)) if neff > 1 else 0.0
    output = args.output or args.data.with_suffix(".sample.spatial.npz" if args.sample_vertices else ".spatial.npz")
    with output.open("wb") as stream:
        np.savez_compressed(stream, bond_ids=bond_ids, edges=edges, atom_ids=atom_ids, coords=coords,
                            box_lengths=box, direction=args.direction, neff=neff,
                            raw_GEBC_igraph=raw_gebc, raw_OBC=raw_obc, raw_DBC=raw_dbc,
                            OBC=raw_obc * norm_factor, DBC=raw_dbc * norm_factor,
                            metadata=json.dumps(metadata))
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
