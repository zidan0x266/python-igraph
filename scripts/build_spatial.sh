#!/usr/bin/env bash
# Run with bash; build both sibling repositories using your chosen python3.
set -euo pipefail

spatial_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
spatial_python="${SPATIAL_PYTHON:-python3}"
spatial_build="$spatial_root/.spatial-build"
spatial_env="$spatial_root/.spatial-venv"
spatial_jobs="${SPATIAL_BUILD_JOBS:-4}"

if [[ ! -x "$spatial_env/bin/python" ]]; then
    "$spatial_python" -m venv "$spatial_env"
fi
"$spatial_env/bin/python" --version
"$spatial_env/bin/python" -m pip install setuptools wheel numpy texttable pytest parameterized

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
        "$spatial_env/bin/python" setup.py build_ext --force
)
IGRAPH_USE_PKG_CONFIG=1 IGRAPH_STATIC=1 PKG_CONFIG_PATH="$spatial_build/install/lib/pkgconfig" \
    "$spatial_env/bin/python" -m pip install --force-reinstall --no-build-isolation --no-deps "$spatial_root/python-igraph"
"$spatial_env/bin/python" -m pytest -q "$spatial_root/python-igraph/tests/test_spatial_betweenness.py" "$spatial_root/python-igraph/tests/test_structural.py"
