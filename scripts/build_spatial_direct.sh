#!/usr/bin/env bash
# Install into the selected Python's own site-packages; no virtualenv is created.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Usage: SPATIAL_PYTHON=/absolute/path/to/python3 bash build_spatial_direct.sh

Builds the sibling modified C igraph and installs python-igraph directly into
this interpreter's site-packages, replacing its installed igraph distribution.
No virtual environment is created. Required build/test dependencies are installed
as needed; unrelated packages such as NetworkX are not targeted.

Optional: SPATIAL_BUILD_JOBS=4 (compilation only)
          SPATIAL_BUILD_DIR=/absolute/path/to/build-directory
Default build directory: <workspace>/.spatial-build-direct
HELP
    exit 0
fi
if [[ $# -ne 0 || -z "${SPATIAL_PYTHON:-}" || "$SPATIAL_PYTHON" != /* || ! -x "$SPATIAL_PYTHON" ]]; then
    echo "Set SPATIAL_PYTHON to the absolute path of your writable Python installation. Use --help for details." >&2
    exit 2
fi

spatial_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
spatial_python="$SPATIAL_PYTHON"
spatial_build="${SPATIAL_BUILD_DIR:-$spatial_root/.spatial-build-direct}"
spatial_jobs="${SPATIAL_BUILD_JOBS:-4}"
if [[ "$spatial_build" != /* || ! "$spatial_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "SPATIAL_BUILD_DIR must be absolute and SPATIAL_BUILD_JOBS must be a positive integer." >&2
    exit 2
fi

# Prevent pip settings from redirecting installation away from this interpreter.
# Keep index/proxy environment settings available for cluster package mirrors.
unset PIP_TARGET PIP_PREFIX PIP_USER
export PIP_CONFIG_FILE=/dev/null
cd "$spatial_root"
"$spatial_python" - <<'PYTHON'
import sys
import sysconfig
import tempfile
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("Use Python 3.11 or newer for the spatial analysis package.")
print("Installing into Python:", sys.executable, flush=True)
print("Python version:", sys.version, flush=True)
for name in ("purelib", "platlib"):
    destination = Path(sysconfig.get_path(name))
    print(f"{name}: {destination}", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=destination):
        pass
if not (Path(sysconfig.get_path("include")) / "Python.h").is_file():
    raise SystemExit("Python development headers are missing (Python.h).")
PYTHON
"$spatial_python" -m pip --version
"$spatial_python" -m pip install --no-user 'setuptools>=64' wheel 'numpy>=1.26' texttable pytest parameterized

cmake -S "$spatial_root/igraph" -B "$spatial_build/core" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$spatial_build/install" \
    -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DIGRAPH_USE_INTERNAL_BLAS=ON -DIGRAPH_USE_INTERNAL_LAPACK=ON \
    -DIGRAPH_USE_INTERNAL_ARPACK=ON -DIGRAPH_USE_INTERNAL_GLPK=ON -DIGRAPH_USE_INTERNAL_GMP=ON \
    -DIGRAPH_GRAPHML_SUPPORT=OFF -DIGRAPH_ENABLE_TLS=ON -DIGRAPH_OPENMP_SUPPORT=OFF
cmake --build "$spatial_build/core" --parallel "$spatial_jobs"
cmake --build "$spatial_build/core" --parallel "$spatial_jobs" \
    --target test_igraph_edge_betweenness_spatial test_igraph_edge_betweenness test_igraph_edge_betweenness_subset
ctest --test-dir "$spatial_build/core" --output-on-failure -R '^test::igraph_edge_betweenness(_spatial|_subset)?$'
cmake --install "$spatial_build/core"

# Use the modified sibling C library, not the uninitialized upstream submodule.
# Force relinking so C-only edits cannot leave a stale statically linked extension.
(
    cd "$spatial_root/python-igraph"
    IGRAPH_USE_PKG_CONFIG=1 IGRAPH_STATIC=1 PKG_CONFIG_PATH="$spatial_build/install/lib/pkgconfig" \
        "$spatial_python" setup.py build_ext --force
)
IGRAPH_USE_PKG_CONFIG=1 IGRAPH_STATIC=1 PKG_CONFIG_PATH="$spatial_build/install/lib/pkgconfig" \
    "$spatial_python" -m pip install --no-user --force-reinstall --no-build-isolation --no-deps "$spatial_root/python-igraph"
"$spatial_python" -m pytest -q "$spatial_root/python-igraph/tests/test_spatial_betweenness.py" "$spatial_root/python-igraph/tests/test_structural.py"

# Check the installed extension, including the raw DBC convention and geometry.
"$spatial_python" - <<'PYTHON'
import igraph as ig
import numpy as np
from igraph import _igraph

assert hasattr(ig.Graph, "edge_betweenness_spatial"), ig.__file__
assert _igraph.SPATIAL_DBC_WEIGHT == "distance"
graph = ig.Graph(n=3, edges=[(0, 1), (1, 2)], directed=False)
coords = np.array([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]])
gebc, obc, dbc = graph.edge_betweenness_spatial(coords, [10., 10., 10.], direction=0)
np.testing.assert_array_equal(gebc, graph.edge_betweenness(directed=False))
np.testing.assert_array_equal(obc, [2., 2.])
np.testing.assert_array_equal(dbc, [6., 6.])
print("Installed in-house igraph:", ig.__file__)
print("Python/C igraph versions:", ig.__version__, ig.__igraph_version__)
print("Verified: import igraph as ig; distance-only raw DBC.")
PYTHON
