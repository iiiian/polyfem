#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$CASE_DIR/../.." && pwd)"
POLYFEM_BIN="$(realpath "${POLYFEM_BIN:-$ROOT_DIR/build/release/PolyFEM_bin}")"
INFLATOR_ROOT="$(realpath "${INFLATOR_ROOT:-$ROOT_DIR/additional_info/microstructure_inflators}")"
INFLATOR_BIN="$INFLATOR_ROOT/build/release/isosurface_inflator/isosurface_cli"
TILE_BIN="$INFLATOR_ROOT/build/release/isosurface_inflator/tile"
PATTERN="$INFLATOR_ROOT/data/patterns/2D/topologies/0105.obj"
RUN_DIR="${1:-$ROOT_DIR/build/shock_protection}"

if (( $# > 1 )); then
    echo "Usage: bash $0 [new-run-directory]" >&2
    exit 2
fi
for executable in "$POLYFEM_BIN" "$INFLATOR_BIN" "$TILE_BIN"; do
    if [[ ! -x "$executable" ]]; then
        echo "Missing executable: $executable" >&2
        exit 2
    fi
done
if [[ ! -f "$PATTERN" ]]; then
    echo "Missing topology: $PATTERN" >&2
    exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "Run directory already exists; choose a new directory: $RUN_DIR" >&2
    exit 2
fi
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
cp "$CASE_DIR/forward.json" "$CASE_DIR/meshing.json" "$CASE_DIR/pin.txt" "$RUN_DIR/"
cd "$RUN_DIR"

echo "Generating topology 0105 in $RUN_DIR"
"$INFLATOR_BIN" 2D_doubly_periodic "$PATTERN" \
    -m meshing.json -S shape-velocities.msh cell.msh > inflation.log 2>&1
"$TILE_BIN" -t 2x2 cell.msh tiled.msh > tiling.log 2>&1
echo "Running incremental compression; log: $RUN_DIR/forward.log"
"$POLYFEM_BIN" -j forward.json > forward.log 2>&1
echo "Output: $RUN_DIR/output/compression.pvd"
