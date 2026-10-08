# Optional inflator library and 3D periodic optimization

Enable the integration explicitly:

```sh
cmake --preset release -DPOLYFEM_WITH_INFLATOR=ON
cmake --build build/release
build/release/tests/unit_tests '[inflator_library],[periodic_contact_3d]'
```

The default is OFF. A disabled build neither downloads nor links the inflator;
selecting `inflated-periodic-shape` then produces a configuration error.

An installed `inflatorConfig.cmake` is preferred when available. Otherwise an
ExternalProject builds the fork in a separate CMake configuration and installs
its shared library and one public header under the PolyFEM build directory.
For local development, configure with
`-DPOLYFEM_INFLATOR_SOURCE_DIR=/absolute/path/to/microstructure_inflators`.
An installed package can be selected with `-Dinflator_DIR=.../lib/cmake/inflator`.
The inflator needs system Boost headers, GMP, and MPFR; its source dependencies
are downloaded automatically. Their targets, include paths, definitions, and
cache settings are not imported into PolyFEM.

## Configuration

The `inflation` object now uses the library, not shell commands. Remove the old
`inflator` and `tiler` executable paths. It contains:

```json
{
  "wire": "cell.obj",
  "symmetry": "orthotropic",
  "work_directory": "mesh-generations",
  "tiles": 1,
  "graph_radius": 2,
  "meshing": {
    "facetSize": 0.1,
    "facetDistance": 0.01,
    "cellSize": 0.2,
    "edgeSize": 0.1
  }
}
```

Also supply the required `initial` array: `[width, ...mesh.parameters]`, using
the parameters returned by `inflator::inflate` for the initial geometry.
Parameter zero remains the x cell width. Other parameters are the inflator's
graph parameters. The normalized cell dimensions are
`[width, 1]` in 2D and `[width, 1, 1]` in 3D; tiling multiplies all dimensions
by `tiles`. Independent optimization of the third cell dimension is not exposed.

Dimension comes from the state's initial mesh. Default `symmetry: "auto"`
selects `doubly_periodic` in 2D and `orthotropic` in 3D. Reflected 3D symmetries
give matching periodic boundary triangulations; unrestricted triply-periodic
CGAL meshing is not supported by this adapter. Supply an initial tetrahedral
mesh from the same graph/symmetry and a 3-by-3 macro displacement gradient.
For bounding-box surface selection, the periodic pairs are `[1,3]`, `[2,4]`,
and `[5,6]`. Optimization states remain static and can use `initial_guess`
for compression-state continuation.

The library returns mesh arrays and graph shape velocities in memory. PolyFEM
applies cell scaling, maps the geometry derivatives into periodic coordinates,
and reconstructs mesh-dependent simulation/cache data. `mesh-generations`
retains mesh and parameter snapshots for diagnosis, not IPC with a subprocess.
Existing 2D preparation scripts still use the standalone CLI to generate their
initial mesh; optimization updates use the library.

## 3D contact and tests

Periodic contact now tiles physical boundary triangles in all three directions,
removes artificial periodic cut faces, stitches seam vertices, and extracts
unique surface edges. Macro-affine displacements are transferred to these
periodic images by the existing contact form.

- The library's own CTest checks 2D/3D generation, periodic matching, stitched
  tiling, deterministic repetition, velocity arrays, and exception propagation.
- `[periodic_contact_3d]` checks a tetrahedral periodic tunnel: cap removal,
  seam stitching, unique edges, active-contact gradient/Hessian finite
  differences, CCD, and a homogenization forward solve.
- `[inflator_library]` checks generated 2D/3D meshes through two compression
  states, adjoint derivatives for width and bar-thickness normal motion, mesh
  regeneration with changed vertex counts, cache identity, and restoration.

The bar-thickness derivative check holds connectivity fixed while perturbing
vertices along the returned shape velocity. It does not claim differentiability
of discrete remeshing. These tests are integration checks, not a reproduction of
the paper's 3D optimized shock-absorption results.

The `Inflator Integration` GitHub Actions workflow downloads the pinned fork,
builds PolyFEM with this option enabled, and runs the focused integration tests.
Run the Linux job locally with:

```sh
act push -W .github/workflows/inflator.yml \
  -P ubuntu-24.04=catthehacker/ubuntu:act-latest
```

The fork's `Library` workflow separately tests GCC and Clang, installation,
an independent consumer, and the public symbol/header boundary. It also tests
the inflator and installed consumer on native Windows MSVC runners. This is
not a Windows test of the full PolyFEM integration. `act` runs the Ubuntu jobs
in Linux containers; it cannot execute native Windows or macOS runners.
No macOS validation has been performed.

## Validation record

The dependency is pinned to `75bb7347b648e800ea90e6822edadbee0946f908`.
Its [native Linux and Windows CI run](https://github.com/iiiian/microstructure_inflators/actions/runs/37729323743)
passes 2D/3D generation and installed-consumer tests. Both Linux compiler jobs
also pass locally under `act`. Local ASan/UBSan tests pass with leak detection.
The focused PolyFEM integration, contact, and schema tests pass 1,566 assertions.

Windows testing exposed an original inflator portability bug: `long(1e12)`
overflowed the 32-bit Windows `long`, making the symmetry tolerance negative.
The pinned revision uses an explicit 64-bit integer denominator. This was
verified with an isolated MSVC reproducer before rerunning mesh tests.
