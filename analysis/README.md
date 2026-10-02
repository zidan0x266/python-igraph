# GEBC, OBC and DBC analysis of LAMMPS networks

This workspace contains a reusable Python package in `python-igraph/analysis/src/spatial_betweenness/`
and a direct command-line entry point, `python-igraph/analysis/src/analyze.py`. It uses the modified
igraph 1.0.0 C/Python libraries in the sibling `igraph/` and `python-igraph/`
repositories. The underlying C routine computes all three channels together
using one unweighted BFS per source, single-threaded, with O(V + E) memory.

## Run with your Python 3.13.15 environment

From this workspace:

```bash
# Inspect either input, including its actual deformed or undeformed box:
.spatial-venv/bin/python python-igraph/analysis/src/analyze.py debug/atrp_s1_sys1_42.data --inspect
.spatial-venv/bin/python python-igraph/analysis/src/analyze.py debug/after_pre.lmp --inspect

# Compute all three centralities for one complete network, loading in x:
.spatial-venv/bin/python python-igraph/analysis/src/analyze.py debug/after_pre.lmp --direction x --validate-stock

# Fast debug run on a smaller induced graph:
.spatial-venv/bin/python python-igraph/analysis/src/analyze.py debug/atrp_s1_sys1_42.data --sample-vertices 256 --validate-stock
```

`--validate-stock` performs a separate stock GEBC pass and fails if custom GEBC
does not agree at `rtol=1e-12, atol=1e-8`. Omit it for routine production after
validation to avoid that second full traversal. Sampling computes a **different
graph** and is not an approximation of the full network's centrality.

Use `--direction x`, `y`, `z`, `0`, `1` or `2`. The default output is
`<input-stem>.<axis>.spatial.npz`, with a matching JSON report. Sample outputs
also contain `.sample<N>`. Supply `--output path/result.npz` to choose a path.
Existing results are protected unless you pass `--overwrite`. Progress goes to
stderr; the JSON summary goes to stdout. `--quiet` suppresses progress logging.

## Python API

Install the lightweight analysis package into the already built environment:

```bash
.spatial-venv/bin/python -m pip install --no-build-isolation --no-deps -e python-igraph/analysis
```

Then use classes from a script or notebook running that environment:

```python
from spatial_betweenness import AnalysisConfig, SpatialBetweennessAnalysis

config = AnalysisConfig(direction="x", validate_stock=True)
analysis = SpatialBetweennessAnalysis.from_lammps("debug/after_pre.lmp", config)
print(analysis.inspect())

result = analysis.run()
raw_gebc, raw_obc, raw_dbc = result.raw_gebc, result.raw_obc, result.raw_dbc
obc, dbc = result.obc, result.dbc
print(result.validation)
result.save("debug/after_pre.x.spatial.npz")
```

After installation, `spatial-betweenness INPUT` and
`python -m spatial_betweenness INPUT` are also available in that environment.
The direct `python python-igraph/analysis/src/analyze.py INPUT` command works without package installation.

For separate input preparation:

```python
from spatial_betweenness import LammpsDataReader, SpatialBetweennessAnalysis

network = LammpsDataReader().read("debug/after_pre.lmp")
analysis = SpatialBetweennessAnalysis(network)
result = analysis.run()
```

## Input and scientific conventions

### Distance-only raw DBC and final postprocessing

Native DBC now accumulates **distance alone**, with pair weight `D_st`. There is
no `Lx`, box-length or bond-length denominator in the C dependency traversal.
`result.raw_dbc` and the saved `raw_DBC` always retain that unscaled sum, in
coordinate length units. It is independent of the OBC loading direction.

Apply any finite positive length afterward, in the same units as the coordinates:

```python
# These all reuse one calculation; result.raw_dbc is never modified.
L_applied = 49.39694662
length_scaled_dbc = result.raw_dbc / L_applied
normalized_dbc = length_scaled_dbc * result.normalization_factor

# Equivalent convenience API:
length_scaled_dbc = result.dbc_for_length(L_applied)
normalized_dbc = result.dbc_for_length(L_applied, normalized=True)
```

You can use the original/undeformed box length, your model's equilibrium bond
length, or an arbitrary reference length. None is inferred automatically.
To obtain the original x box length from a data file, use
`LammpsDataReader().read("debug/after_pre.lmp").box_lengths[0]`.

For convenience, `--dbc-length L` or `AnalysisConfig(dbc_length=L)` selects the
length for **postprocessed `DBC` only**; the raw array always stays unscaled:

```bash
# Compute distance-only raw DBC, and save DBC with an undeformed length applied:
.spatial-venv/bin/python python-igraph/analysis/src/analyze.py debug/atrp_s1_sys1_42.data --dbc-length 49.39694662 --output debug/deformed_reference_L0.npz

# Example for a model whose equilibrium bond length is 1.0:
.spatial-venv/bin/python python-igraph/analysis/src/analyze.py debug/atrp_s1_sys1_42.data --dbc-length 1.0 --output debug/deformed_reference_bond.npz
```

Without `--dbc-length` (or with `dbc_length=None`), `result.dbc` / saved `DBC`
contains only pair normalization: `raw_dbc * 2 / (neff * (neff - 1))`.
With a length, it contains `(raw_dbc / L) * 2 / (neff * (neff - 1))`.
GEBC and OBC are unaffected. The actual simulation box is still used for
minimum-image distances, even when postprocessing uses a different length.

New output uses `schema_version=2` and `raw_dbc_weight="distance"`. JSON metadata
records `dbc_length=null` when none is selected. The NPZ stores the postprocessing
divisor (`dbc_length=1.0` when none is selected) and a `dbc_length_applied` boolean.
Use distinct `--output` paths when comparing different processed outputs.

**Migration:** older saved `raw_DBC` used weight `D_st / L_old`; those files have
not been overwritten. Recover the distance-only sum with
`raw_dbc_distance = old_raw_dbc * L_old`, then divide by any new length.
Do not apply the new formula directly to an old length-divided array. In old
files `L_old` is the selected `dbc_length`, or the loading-axis box length when
no custom denominator was selected.

Both native libraries must be rebuilt for this convention. The analysis API
checks `igraph._igraph.SPATIAL_DBC_WEIGHT == "distance"` and rejects the older
extension instead of silently treating length-divided data as distance-only.

### Network conventions

- Input is a LAMMPS **data file**, with `Atoms # full` rows
  `id molecule type charge x y z [ix iy iz]` and `Bonds` rows
  `bond_id bond_type atom1 atom2`. A bare `Atoms` header assumes full style.
  Other atom styles and trajectory/dump files are not supported.
- Orthorhombic bounds come from `xlo xhi`, `ylo yhi`, `zlo zhi` in each file.
  Periodicity is assumed along **all three axes**; a data file does not encode
  the simulation's boundary settings. Triclinic cells are rejected.
- Coordinates are sorted by positive, unique atom ID. Nonconsecutive IDs are
  supported. Bond rows and endpoint orientation retain file order. The reader
  checks counts, finite geometry, valid endpoints and unique bond IDs. Analysis
  rejects self-loops and duplicate atom-pair bonds.
- The graph remains at bead/bond level. GEBC uses pair weight `1`, OBC uses
  `abs(minimum_image_displacement[direction]) / r`, and DBC uses
  `r` without a reference-length denominator.
  Coincident pairs have zero spatial weight.
- All raw arrays already follow stock igraph's undirected convention. Do **not**
  halve them again. GEBC is left raw. Normalized OBC/DBC use
  `2 / (neff * (neff - 1))`, where `neff` counts vertices with nonzero degree.
  Empty/edgeless networks return empty edge arrays without division by zero.
- GEBC, OBC and DBC always share the same unweighted paths. This package never
  passes geometry through igraph's edge-length `weights=` argument.

## Output mapping

```python
import numpy as np

with np.load("debug/after_pre.x.spatial.npz", allow_pickle=False) as data:
    bond_ids = data["bond_ids"]
    endpoint_atom_ids = data["bond_atom_ids"]
    raw_gebc = data["raw_GEBC_igraph"]
    raw_obc, raw_dbc = data["raw_OBC"], data["raw_DBC"]
    obc, dbc = data["OBC"], data["DBC"]
```

Every row `i` refers to the same original bond in all these arrays. `edges`
contains zero-based **coordinate-row indices**, while `bond_atom_ids` contains
actual LAMMPS atom IDs: `atom_ids[edges] == bond_atom_ids`. For consecutive IDs in
a full network, `edges` is the original atom endpoint IDs minus one. For
nonconsecutive IDs or samples, use the explicit mapping instead.

The NPZ also includes `atom_ids`, `coords`, `bond_types`, `box_bounds`,
`box_lengths`, `direction`, `dbc_length`, `dbc_length_applied`, `raw_dbc_weight`,
`schema_version`, `neff` and JSON `metadata`. The separate JSON file
records versions, timing, box, sample status and optional GEBC validation.
Peak RSS is the process lifetime high-water mark through the calculation,
including Python/input arrays; it is not an isolated measurement of native memory.

## Source organization and tests

| File | Responsibility |
| --- | --- |
| `python-igraph/analysis/src/analyze.py` | Direct runnable entry point |
| `python-igraph/analysis/src/spatial_betweenness/models.py` | `Direction`, `AnalysisConfig`, `PolymerNetwork` |
| `python-igraph/analysis/src/spatial_betweenness/lammps.py` | Streaming `LammpsDataReader` |
| `python-igraph/analysis/src/spatial_betweenness/analysis.py` | `SpatialBetweennessAnalysis`, native call and validation |
| `python-igraph/analysis/src/spatial_betweenness/results.py` | `CentralityResult`, normalization and NPZ/JSON writing |
| `python-igraph/analysis/src/spatial_betweenness/cli.py` | Arguments, logging and error reporting |
| `python-igraph/analysis/tests/test_spatial_analysis.py` | Analytical, path-enumeration, parser and CLI tests |

```bash
.spatial-venv/bin/python -m pytest -q -c python-igraph/analysis/pyproject.toml python-igraph/analysis/tests
```

The package uses type hints, dataclasses, `pathlib`, logging, explicit validation
and a standard `src` packaging layout. Numerical arrays in the input/result
records are read-only. The library API leaves global igraph progress handlers
alone; the standalone CLI manages its own progress callback.

The stock PyPI igraph package does not contain `edge_betweenness_spatial`.
On a new machine, first build the modified C and Python libraries with
`bash python-igraph/scripts/build_spatial.sh`; then install this analysis package.
See [the native modification guide](../SPATIAL_BETWEENNESS.md) for
the build details and original validation measurements. Rebuild on Bebop/Linux;
the local macOS binaries cannot be copied there.

This directory is the version-controlled analysis package. Commands above run from the workspace containing both sibling repositories and `debug/`.
The earlier `python-igraph/scripts/spatial_lammps.py` remains available for
compatibility; new analyses can use the classes and entry points above.

## Compare the old and new DBC conventions

The comparison script verifies input/bond mapping, recomputes all channels and
checks new raw DBC against the old raw DBC multiplied by its original length.
It also recovers old raw/normalized DBC by division, tries multiple new lengths,
and checks that GEBC/OBC are unchanged. Add `--validate-stock` for a fresh stock
GEBC pass. Existing reference files are read-only inputs.

```bash
.spatial-venv/bin/python python-igraph/analysis/src/compare_dbc_conventions.py debug/atrp_s1_sys1_42.data debug/atrp_s1_sys1_42.spatial.npz --output debug/new_distance_only.npz --validate-stock
```

A `.comparison.json` report contains all differences, correlations and tolerances.
The algebraic relation is exact; floating-point summation and division can differ
in their final rounding. The report also records exact array equality explicitly.
The new result archive is saved only when every comparison passes.

## Full-network DBC convention comparison (2026-10-02)

The 103,149-node / 105,869-edge input was recomputed with distance-only native
DBC and compared with the saved distance/Lx calculation, using
Lx = 153.13053451563727. Input coordinates, box, atom IDs, bond IDs and edge
order matched exactly. The mathematical identity is `DBC(L) = raw_DBC / L`
for any positive constant L; floating-point evaluation order can change rounding.

| Comparison | Maximum absolute difference | Maximum relative difference | Exact array equality |
| --- | --- | --- | --- |
| GEBC versus previous saved GEBC | 0 | 0 | True |
| OBC versus previous saved OBC | 0 | 0 | True |
| New raw DBC / Lx versus previous raw DBC | 1.0766088962554932e-6 | 6.150020820066422e-14 | False |
| Recovered normalized DBC versus previous normalized DBC | 2.0296264668928643e-16 | 6.15462511637816e-14 | False |
| New GEBC versus fresh stock igraph | 1.862645149230957e-9 | 2.9295008461326823e-16 | False |

All comparisons passed `allclose(rtol=1e-12, atol=1e-8)`. Lengths 1, 2.5 and
49.39694662 also passed the rescaling comparison. This checks the denominator
change against the previous implementation, not an independent full-network
OBC/DBC reference. Native spatial runtime was 684.94 s; stock GEBC was 489.28 s
on this local macOS machine with Python 3.13.15. These are not Bebop timings.

See the [complete comparison report](validation/dbc_distance_comparison.json).
