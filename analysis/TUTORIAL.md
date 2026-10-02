# Installing in-house igraph and analyzing polymer networks

This tutorial builds the modified C igraph and python-igraph 1.0.0 libraries,
installs the analysis package, and computes GEBC, OBC and DBC from a LAMMPS
data file. Run all commands from the workspace containing both repositories
unless a step explicitly changes directory.

The native function computes all three centralities together using one
unweighted shortest-path traversal per source. Raw DBC contains distance alone;
you can choose its reference length after the calculation.

## 1. Prepare your Python and build tools

You can use your Python 3.13.15. The complete build and analysis have been tested
locally with that version on macOS. On Bebop, compile again using the Linux
compiler and Python available there; do not copy the macOS environment or binaries.

You need Git, Python with `venv` and development headers, a C/C++ compiler,
CMake 3.18 or newer, Make, Bison, Flex and pkg-config. On Bebop, load the site
modules providing these tools before building. Module names depend on the site
configuration, so this tutorial does not assume specific names.

Check the tools in your current shell:

```bash
python3 --version
git --version
cc --version
c++ --version
cmake --version
make --version
bison --version
flex --version
pkg-config --version
```

The build script downloads Python build/test dependencies through pip. It
installs into your workspace and does not require a system-wide installation.
Use a machine or cluster node where those downloads and compilation are allowed.

## 2. Get both modified repositories

For a new installation:

```bash
mkdir -p "$HOME/spatial-igraph-workspace"
cd "$HOME/spatial-igraph-workspace"
git clone --branch spatial-betweenness-1.0.0 https://github.com/zidan0x266/igraph.git
git clone --branch spatial-betweenness-1.0.0 https://github.com/zidan0x266/python-igraph.git
mkdir -p debug results
```

If you already have the two clones, use their existing parent directory instead.
Check `git -C igraph status` and `git -C python-igraph status` first. With no
uncommitted work, switch and update them:

```bash
git -C igraph switch spatial-betweenness-1.0.0
git -C python-igraph switch spatial-betweenness-1.0.0
git -C igraph pull --ff-only origin spatial-betweenness-1.0.0
git -C python-igraph pull --ff-only origin spatial-betweenness-1.0.0
mkdir -p debug results
```

The layout must be:

```text
spatial-igraph-workspace/
  igraph/
  python-igraph/
    analysis/
    scripts/build_spatial.sh
  debug/
  results/
```

Both repositories must use the modified branch. The build uses the sibling
`igraph/` source directly; initializing python-igraph's stock C submodule is
unnecessary. The input data and large result archives are not in these GitHub
repositories. Copy your LAMMPS data file into `debug/` separately.

## 3. Build the native libraries

To use the `python3` found in your current shell:

```bash
SPATIAL_BUILD_JOBS=4 bash python-igraph/scripts/build_spatial.sh
```

Alternatively, select a specific Python when creating a new environment:

```bash
SPATIAL_PYTHON=/absolute/path/to/python3.13 SPATIAL_BUILD_JOBS=4 bash python-igraph/scripts/build_spatial.sh
```

Run one of these commands. The script creates `.spatial-venv/`, builds and tests
the C library in `.spatial-build/core/`, installs it in `.spatial-build/install/`,
and builds/tests the Python extension against that library. It forces relinking
so that changes to the statically linked C code are included.

`SPATIAL_BUILD_JOBS` controls compilation only. Centrality calculation is
single-threaded. This build disables GraphML support and OpenMP.

An existing `.spatial-venv` retains its original Python interpreter even if you
change `SPATIAL_PYTHON`. Use a fresh workspace if you want a separate build with
a different Python version.

Activate the environment and install the analysis package:

```bash
source .spatial-venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e python-igraph/analysis
```

The build script already installs the package's NumPy dependency. `--no-deps`
keeps this installation from resolving another igraph distribution. The ordinary
PyPI igraph package does not provide the in-house spatial method.

In each new terminal, return to this workspace and activate `.spatial-venv`
again. You can also use `.spatial-venv/bin/python` explicitly without activation.

## 4. Verify the installation

Run this from the workspace, outside the `python-igraph/` source directory:

```bash
python - <<'PY'
import sys
import igraph as ig
import numpy as np
from igraph import _igraph

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Python igraph:", ig.__version__)
print("C igraph:", ig.__igraph_version__)
print("Imported from:", ig.__file__)
assert hasattr(ig.Graph, "edge_betweenness_spatial")
assert _igraph.SPATIAL_DBC_WEIGHT == "distance"

graph = ig.Graph(n=3, edges=[(0, 1), (1, 2)], directed=False)
coords = np.array([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]])
gebc, obc, dbc = map(np.asarray, graph.edge_betweenness_spatial(coords, [10., 10., 10.], direction=0))
np.testing.assert_array_equal(gebc, [2., 2.])
np.testing.assert_array_equal(obc, [2., 2.])
np.testing.assert_array_equal(dbc, [6., 6.])
np.testing.assert_array_equal(gebc, graph.edge_betweenness(directed=False))
print("Installation verified; raw DBC:", dbc)
PY

python -m pytest -q -c python-igraph/analysis/pyproject.toml python-igraph/analysis/tests
python -m spatial_betweenness --help
```

Python igraph should report `1.0.0`. The C version may include a Git revision
suffix after `1.0.0`. The method and `SPATIAL_DBC_WEIGHT == "distance"` checks
confirm that you imported the required modified extension.

## 5. Inspect your LAMMPS input

The remaining examples use `debug/atrp_s1_sys1_42.data`. Substitute your own
filename, including `.lmp` if appropriate; the contents determine its format.

The reader expects a LAMMPS **data file**, not a trajectory dump:

- `Atoms # full`: `id molecule type charge x y z`, with optional image flags.
- `Bonds`: `bond_id bond_type atom1 atom2`.
- Orthorhombic `xlo xhi`, `ylo yhi`, `zlo zhi` bounds.

The analysis assumes periodicity in all three directions. It rejects triclinic
cells, self-loops and duplicate atom-pair bonds. It retains the bead/bond graph,
supports nonconsecutive atom IDs, and preserves the original bond-row ordering.

```bash
python -m spatial_betweenness debug/atrp_s1_sys1_42.data --inspect
```

This validates the input and prints the topology, actual box and analysis
settings without running all-source centrality. Confirm the node and bond
counts and box lengths before proceeding.

## 6. Run a small validation calculation

```bash
python -m spatial_betweenness debug/atrp_s1_sys1_42.data --direction x --sample-vertices 256 --validate-stock --output results/sample256.x.npz
```

This computes a 256-vertex induced neighborhood and compares its custom GEBC
against stock igraph. It is a debugging calculation on a different graph, not
an estimate of full-network centrality.

Successful runs create an NPZ array archive and a matching JSON report. Look for
`gebc_validation.allclose: true` in the report. Validation uses
`rtol=1e-12, atol=1e-8`; full-network floating-point results may differ from stock
by roundoff even though the mathematical quantity is identical.

## 7. Analyze the complete network

For x loading, with stock GEBC validation and an undeformed DBC reference length
of 49.39694662 in the same units as your coordinates:

```bash
python -m spatial_betweenness debug/atrp_s1_sys1_42.data --direction x --dbc-length 49.39694662 --validate-stock --output results/network.x.npz
```

Replace the example length with the one appropriate for your system. The
reference length affects only postprocessed DBC. Minimum-image distances always
use the actual box in the input file, including its deformation.

Use `--direction y` or `--direction z` for another OBC loading direction. GEBC
and distance-only raw DBC are independent of that direction. Changing OBC's
direction requires another calculation with the current API.

The outputs are:

```text
results/network.x.npz
results/network.x.json
```

`--validate-stock` runs a separate stock GEBC calculation after the combined
native calculation. For routine production after validation, omit it to avoid
that additional full pass. On Bebop, run full analyses within a compute allocation
using the site's scheduler and account settings. One analysis uses one CPU;
requesting multiple CPUs does not parallelize this implementation.

Existing outputs are protected. Choose a new `--output` filename for another
run, or use `--overwrite` when replacement is intended. Progress is printed to
stderr, and the final JSON summary to stdout.

## 8. Understand the saved arrays

Let `f = 2 / (neff * (neff - 1))`, where `neff` counts atoms with nonzero degree.
The implementation uses `f = 0` when fewer than two such atoms exist.

| NPZ key | Meaning |
| --- | --- |
| `bond_ids` | Original LAMMPS bond IDs, in file row order |
| `bond_atom_ids` | Original endpoint atom IDs for each bond |
| `raw_GEBC_igraph` | Raw undirected GEBC, pair weight 1 |
| `raw_OBC` | Raw undirected OBC, pair weight `abs(dr_loading) / r` |
| `raw_DBC` | Raw undirected DBC, pair weight `r`, with no length divisor |
| `OBC` | `raw_OBC * f` |
| `DBC` | `(raw_DBC / L) * f` when `--dbc-length L` is supplied; otherwise `raw_DBC * f` |

Every entry at index `i` refers to the same bond across these arrays. `edges`
contains zero-based coordinate-row indices; `bond_atom_ids` contains the actual
LAMMPS IDs. Raw arrays already include igraph's undirected correction: do not
apply another factor of one half. GEBC is left raw.

Without a reference length, raw DBC and pair-normalized DBC retain coordinate
length units. Dividing by a length in those same units makes DBC dimensionless.

## 9. Change the DBC length without rerunning centrality

Save the following as `postprocess_dbc.py` in the workspace, then run
`python postprocess_dbc.py`. It reads the archive from step 7 and creates a new
archive containing three DBC choices and their bond mapping:

```python
from pathlib import Path
import numpy as np

source = Path("results/network.x.npz")
destination = Path("results/network.x.dbc_lengths.npz")
if destination.exists():
    raise FileExistsError(destination)

with np.load(source, allow_pickle=False) as data:
    if "raw_dbc_weight" not in data or str(data["raw_dbc_weight"].item()) != "distance":
        raise ValueError("Expected a new distance-only raw DBC archive")
    raw_dbc = data["raw_DBC"]
    neff = int(data["neff"].item())
    factor = 2.0 / (neff * (neff - 1)) if neff > 1 else 0.0
    arrays = {"bond_ids": data["bond_ids"], "bond_atom_ids": data["bond_atom_ids"]}

    # Replace these example lengths with your scientific choices.
    lengths = {"undeformed": 49.39694662, "equilibrium_bond": 1.0, "arbitrary": 2.5}
    for label, length in lengths.items():
        if not np.isfinite(length) or length <= 0:
            raise ValueError(f"Invalid reference length: {length}")
        arrays[f"L_{label}"] = np.asarray(length)
        arrays[f"DBC_length_scaled_{label}"] = raw_dbc / length
        arrays[f"DBC_normalized_{label}"] = (raw_dbc / length) * factor

np.savez_compressed(destination, **arrays)
print(destination)
```

This performs only array postprocessing. The same raw DBC supports any positive
constant reference length; no graph traversal is needed. The algebraic identity
is exact, although dividing after accumulation can round differently from
dividing each pair contribution before accumulation.

Older archives used `D_st / L_old` as their raw DBC weight. To migrate those,
first recover `raw_dbc_distance = old_raw_dbc * L_old`. Do not treat an old archive
as distance-only. See the [convention comparison report](validation/dbc_distance_comparison.json)
for the measured old/new agreement on the full network.

## 10. Use the class API in your own script

This is an alternative to the CLI calculation in step 7:

```python
from spatial_betweenness import AnalysisConfig, SpatialBetweennessAnalysis

config = AnalysisConfig(direction="x", validate_stock=True, dbc_length=49.39694662)
analysis = SpatialBetweennessAnalysis.from_lammps("debug/atrp_s1_sys1_42.data", config)
print(analysis.inspect())
result = analysis.run()

raw_gebc = result.raw_gebc
raw_obc = result.raw_obc
raw_dbc = result.raw_dbc
obc = result.obc
dbc = result.dbc

# Reuse the same calculation for another reference length.
dbc_at_bond_length = result.dbc_for_length(1.0, normalized=True)
print(result.validation)
result.save("results/network.api.x.npz")
```

For a notebook, select the Python interpreter from this workspace's
`.spatial-venv/bin/python`. The [analysis README](README.md) describes the
classes and output metadata in more detail.

## Troubleshooting and later updates

| Symptom | Action |
| --- | --- |
| No `edge_betweenness_spatial` method | Check `sys.executable` and `igraph.__file__`; activate the built environment and confirm both branch names. |
| Missing or incorrect `SPATIAL_DBC_WEIGHT` | Rebuild both libraries with `build_spatial.sh`; an older extension may still contain length-divided DBC. |
| CMake cannot find a compiler, Bison, Flex or pkg-config | Load/install the missing build tool, then rerun the build script. |
| Missing `Python.h` | Make the development headers for the selected Python available before rebuilding. |
| Output already exists | Choose a different `--output` or intentionally use `--overwrite`. |
| Unsupported atom style or triclinic box | Supply a supported full-style, orthorhombic data file; do not merely relabel incompatible data. |
| Custom GEBC validation fails | Preserve the error log and investigate before using OBC/DBC scientifically. |

When you later update the branches, run the two `git pull --ff-only` commands
from step 2, rerun `build_spatial.sh`, and repeat the installation checks.
Updating source alone does not update the compiled extension. You can record
the exact source versions used for a run with:

```bash
git -C igraph rev-parse HEAD
git -C python-igraph rev-parse HEAD
```

The [native modification guide](../SPATIAL_BETWEENNESS.md) explains the C/Python
changes and validation history. Local benchmark times in that guide do not
establish performance on Bebop.
