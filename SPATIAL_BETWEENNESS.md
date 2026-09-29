# Native spatial edge betweenness (in-house igraph 1.0.0)

The new `Graph.edge_betweenness_spatial(coords, box_lengths, direction=0)` returns
`(raw_gebc, raw_obc, raw_dbc)` as three Python lists in edge-ID order, following
igraph's usual return convention. Convert with `np.asarray` for NumPy operations.
Stock `Graph.edge_betweenness()` is unchanged.

For each source, this calls the existing C `sspf_edge()` once, then accumulates
three dependency channels in the same reverse traversal. GEBC has pair weight 1;
OBC has `abs(dr[direction]) / r` (zero when `r == 0`); DBC has
`r / box_lengths[direction]`. Each displacement uses
`dr -= length * nearbyint(dr / length)`. The loading direction is 0=x, 1=y, 2=z.
All outputs use igraph's raw undirected convention, with exactly one division by
two. Disconnected pairs contribute nothing. Coordinates do not change shortest
paths. The implementation is single-threaded and uses O(V + E) memory.

The native API supports undirected graphs, including loops (zero contribution)
and parallel edges. Coordinates must have shape `(vcount, 3)` with finite values;
box lengths must be three finite positive values. Triclinic cells are unsupported.
The LAMMPS runner additionally checks that the production graph is simple.

## Source layout and release bases

Keep the two modified repositories as siblings:

```text
workspace/
  igraph/
  python-igraph/
  debug/
```

Both development branches are named `spatial-betweenness-1.0.0` and start at the
official `1.0.0` tags. Python's release submodule points to the same C release.
The build script uses the modified sibling C repository through `pkg-config`,
so initializing Python's stock C submodule is unnecessary.

Files implementing the extension:

- C: `igraph/src/centrality/betweenness.c`, `igraph/include/igraph_centrality.h`.
- C documentation/test registration: `igraph/doc/structural.xxml`, `igraph/tests/CMakeLists.txt`.
- C test: `igraph/tests/unit/igraph_edge_betweenness_spatial.c`.
- Python binding and method table: `python-igraph/src/_igraph/graphobject.c`.
- Python validation: `python-igraph/tests/test_spatial_betweenness.py`.

No C source-list edit is needed: `betweenness.c` is already compiled. The new
declaration uses `IGRAPH_EXPORT`, and the normal install exports the public
header. The Python extension statically links the local C library.

## Build with your Python

Prerequisites: Python 3.13.15 (or another supported Python), a C/C++ toolchain,
CMake >= 3.18, Make, Bison, Flex and pkg-config. On Bebop, load the corresponding
site modules before building. No root installation is needed.

From the workspace directory:

```bash
bash python-igraph/scripts/build_spatial.sh
```

This uses `python3` to create `.spatial-venv`, installs build/test dependencies
there, builds C in `.spatial-build/core`, installs it in `.spatial-build/install`,
and builds/tests the Python extension against it. To choose a Python executable
for a **new** environment, set `SPATIAL_PYTHON=/path/to/python3`. An existing
environment retains the Python interpreter with which it was created.
`SPATIAL_BUILD_JOBS=4` controls compilation only; centrality remains single-threaded.
GraphML support is disabled in this isolated build to avoid requiring libxml2.

Use `.spatial-venv/bin/python` or activate the environment:

```bash
source .spatial-venv/bin/activate
python -c 'import igraph; print(igraph.__version__, igraph.__igraph_version__, hasattr(igraph.Graph, "edge_betweenness_spatial"))'
```

The expected versions are `1.0.0 1.0.0` and the method check must be `True`.
Rebuild separately on Linux/Bebop; macOS binaries are not portable to it. Clone
both development branches into the same workspace before building:

```bash
git clone --branch spatial-betweenness-1.0.0 https://github.com/zidan0x266/igraph.git
git clone --branch spatial-betweenness-1.0.0 https://github.com/zidan0x266/python-igraph.git
bash python-igraph/scripts/build_spatial.sh
```

The LAMMPS input and generated validation archives live outside these repositories;
copy them separately if needed for a Bebop comparison.

## Use in analysis

```python
import igraph as ig
import numpy as np

G = ig.Graph(n=len(coords), edges=edges, directed=False)
raw_gebc, raw_obc, raw_dbc = [np.asarray(v) for v in G.edge_betweenness_spatial(coords, box_lengths, direction=0)]
neff = np.count_nonzero(G.degree())
norm_factor = 2.0 / (neff * (neff - 1)) if neff > 1 else 0.0
obc = raw_obc * norm_factor
dbc = raw_dbc * norm_factor
```

GEBC stays raw. Do not halve native outputs again. When comparing with an old
reference that sums **ordered** source-target pairs without halving, divide that
reference by two first. Do not infer its convention from the label “raw”.

## Run the LAMMPS file

The runner reads `Atoms # full` and `Bonds`, checks consecutive atom IDs, reads
the actual orthorhombic box, preserves bond-row order, and verifies igraph's
edge-ID order. It keeps the bead-level graph; it does not contract polymer strands.
Periodic image flags are unnecessary because spatial distances use minimum image.

```bash
# Validate the entire input and edge order without doing all-source centrality:
.spatial-venv/bin/python python-igraph/scripts/spatial_lammps.py debug/atrp_s1_sys1_42.data --inspect

# Fast debug calculation on a 256-vertex induced neighborhood:
.spatial-venv/bin/python python-igraph/scripts/spatial_lammps.py debug/atrp_s1_sys1_42.data --sample-vertices 256 --validate-stock

# Full calculation, followed by a separate stock GEBC validation pass:
.spatial-venv/bin/python python-igraph/scripts/spatial_lammps.py debug/atrp_s1_sys1_42.data --validate-stock
```

The sample is a different, smaller graph; its centralities are **not** an estimate
of full-network values. Full production calculation has no source chunking,
multiprocessing, NetworKit or Python dependency accumulation. Omit
`--validate-stock` after validation to avoid the second all-source traversal.

Outputs are `<input>.spatial.npz` and a JSON timing/validation report. Use
`--output PATH.npz` to change the destination. The archive contains `bond_ids`,
`edges` (original zero-based atom endpoints), `raw_GEBC_igraph`, `raw_OBC`,
`raw_DBC`, `OBC`, `DBC`, coordinates, atom IDs, box lengths and metadata.
`atom_ids` identifies coordinate rows, including in debug subgraphs. Existing
outputs at the selected destination are overwritten.

The supplied file has 103,149 atoms and 105,869 bonds. Its deformed box is
`[153.13053451563727, 28.085726639308756, 28.085726639308756]`; use these lengths,
not the undeformed cubic lengths. Loading defaults to x.

## Validation scope

Tests compare native GEBC directly with stock and OBC/DBC with independently
enumerated unordered-pair shortest paths on small graphs. Cases cover all three
axes, multiple shortest paths, periodic wrapping, coincident vertices,
disconnected/empty graphs, parallel edges, loops, NumPy views, edge ordering and
invalid inputs. C tests also check error handling and the cleanup stack.

The full-file runner records maximum/mean absolute GEBC differences, maximum
relative difference (denominator floor `1e-15`), Pearson correlation and
`allclose(rtol=1e-12, atol=1e-8)`. It refuses to save results if GEBC validation
fails or native outputs are nonfinite. Its peak RSS is for the entire process,
including input and Python arrays, not just the native traversal.

The user's existing NetworKit/Python OBC/DBC reference and its full-network
outputs were not supplied; comparison against those remains separate from the
independent small-graph tests. Local macOS timings do not establish Bebop speedup.
OpenMP is intentionally deferred.

## Measured local result (2026-09-28)

On macOS arm64 with Python 3.13.15, the complete supplied 103,149-node /
105,869-edge network passed stock GEBC validation for x loading:

| Check | Result |
| --- | --- |
| Maximum absolute GEBC difference | `1.862645149230957e-9` |
| Mean absolute GEBC difference | `2.0507851704203632e-13` |
| Maximum relative GEBC difference | `2.9295008461326823e-16` |
| Pearson correlation | `1.0` |
| `allclose(rtol=1e-12, atol=1e-8)` | `True` |
| Combined GEBC/OBC/DBC runtime | `708.49 s` (11.81 min) |
| Stock GEBC runtime, same build/machine | `509.80 s` (8.50 min) |
| Peak process RSS through both passes | `70.64 MiB` |

The three-channel calculation took 1.39 times the stock GEBC-only runtime in
this run. These are local measurements, not Bebop timings or a comparison with
the previous multiprocessing workflow. GEBC agrees within floating-point
roundoff; it is not bitwise identical on this full graph.

On the 256-vertex / 257-edge induced sample, independent enumeration of all
unordered source-target shortest paths passed for all three loading directions.
For x loading, the maximum absolute differences were `9.095e-11` for OBC and
`6.821e-12` for DBC, with maximum relative differences below `1e-14`.

Full results are in `debug/atrp_s1_sys1_42.spatial.npz` beside the repositories.
The matching `.json` records timing and GEBC validation; the sample's
`.sample.reference.json` records the independent OBC/DBC comparison. The saved
full archive was checked against the original bond IDs, endpoints, coordinates
and box, and its normalized arrays were verified against the requested formula.
All 105,869 entries in each channel are finite and nonnegative.
