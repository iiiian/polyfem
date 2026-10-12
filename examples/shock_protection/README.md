# Shock-protecting microstructure

Forward compression and shape optimization of microstructures from Huang et al.,
*Optimized shock-protecting microstructures* (2024), using the VarForm CLI.
The default topology `0105` is the quick-start example in the authors' `ShockProtection`
repository. Both runners start from **unoptimized geometry**, not a published
optimized design.

## Shape optimization

Requires a Release PolyFEM build with optimization, CHOLMOD, and
`POLYFEM_WITH_INFLATOR=ON`. Optimization updates call the isolated inflator
shared library. The preparation script still uses separately built
`isosurface_cli` and `tile` executables for its initial mesh. See
[the library integration guide](INFLATOR_LIBRARY.md) for the API, build options,
and 3D support.

```bash
cmake --preset release -DPOLYFEM_WITH_INFLATOR=ON
cmake --build build/release --target PolyFEM_bin unit_tests
cmake --build additional_info/microstructure_inflators/build/release --target isosurface_cli tile
python3 examples/shock_protection/optimize.py build/shock_optimization
```

The runner refuses to overwrite an existing directory. `--inflator-root` and
`--polyfem` override the external source tree and PolyFEM executable paths.
`--prepare-only` writes the configuration without starting optimization.
Neither runner downloads or builds dependencies.

### Smaller optimization example

Select an inflator graph with `--topology`. The diamond lattice `0004` has
23 optimization variables including width, versus 51 for `0105`. With the
default meshing settings and initial parameters, its 2×2 tile has 4,384
triangles versus 9,828 for `0105`. Keep the 2×2 tile to retain multi-cell
deformation modes; fewer elements do not guarantee a particular solver speedup.

```bash
~/miniconda3/envs/sc/bin/python examples/shock_protection/optimize.py \
  build/shock_0004 --topology 0004 --strains .10 .15 .20 .25 --iterations 3
```

This is a limited-budget workflow check, not a converged optimization. Omit
`--iterations 3` for the default 100-iteration budget. The initial diamond
outline visually matches Figure 37's fourth row, but the figure does not
identify its graph parameterization or optimized parameter vector. The default
5000 Pa target is a test target, not an identified target for that figure.
When extending a saved run, pass the same `--topology` along with `--parameters`.

### Figure 20 A-to-B comparison

See [figure20/README.md](figure20/README.md) for the reconstructed extended-chi
baseline, shape optimization, frozen validated parameters, and independent
comparison against the paper's vector stress curves and geometry images.
The refined B response agrees to 2.39% relative RMS error over 10–60%, but
the geometry and small-strain stiffness differ; the final strict 5%
optimization-sample criterion is not met. The report records these limitations.

### Default configuration

The optimization uses:

- A stitched 2×2 simulation tile and three static homogenization states at
  10%, 15%, and 20% compression. Each state initializes its displacement and G
  from the preceding strain state, on the same regenerated mesh.
- 51 variables: cell width, 22 graph-position parameters, 14 radii, and 14
  blending parameters. Initial width is 1, radii are 0.05, and blending is 0.01,
  following the authors' optimization script rather than the inflator's 0.07
  default thickness used by the forward example.
- Target compressive stress 5000 Pa. At each strain the objective is
  (−∫material Pyy dA / (5000 × 4 × width) − 1)² + 50 Gxy² + 50 Gxx².
  P is first Piola stress. The denominator uses the **whole reference tile
  area**, including voids; it changes with cell width.
- Bound-constrained L-BFGS, with the authors' parameter bounds and maximum
  changes. One optimizer instance survives all mesh regenerations: the
  51-dimensional variables and optimizer history do not change size.

The optimization runner passes `--ns` because the JSON spec engine cannot
strictly validate recursive objective trees. Use `--ns` when launching the
generated `opt.json` manually as well.

For every new parameter vector, `inflated-periodic-shape` calls the library,
receives the tiled mesh and shape velocities in memory, scales the mesh, replaces each
VarForm's mesh-dependent data, and clears its differentiation cache. The
VarForm and cache objects themselves retain their identities, so objectives
remain attached to the current simulation. FE spaces, periodic reductions,
contact forms, and adjoint systems are rebuilt before evaluation. No legacy
State solve is used.

Each equilibrium is differentiated implicitly. The preceding state's
solution is an initial guess, not a physical state variable differentiated
through the loading sequence. Meshes may change connectivity and vertex count
between optimization evaluations; continuation between strain states requires
identical meshes and FE spaces. Dynamic optimization, SLIM, and interpolation
between different meshes are not supported by this integration.

Outputs:

- `parameters.txt`: accepted parameter vectors, including iteration zero.
- `state-*/opt_state_*_iter_*.vtu`: equilibria at accepted iterations.
- `state-*/opt_state_*_iter_*.obj`: their reference meshes.
- `mesh-generations/<index>/`: every trial mesh, parameters, velocity fields,
  and external-program logs, including rejected line-search trials.
- `optimization.log`: solver output and `stress_ratio_<strain>` values.

`--iterations` limits the optimization budget; reaching it is not evidence of
convergence. To extend the strain range using a saved design:

```bash
python3 examples/shock_protection/optimize.py build/shock_optimization_25 \
  --parameters build/shock_optimization/parameters.txt --strains .10 .15 .20 .25
```

This starts a new optimization for the extended objective. Within each run,
mesh regeneration does not restart the optimizer. The paper starts at 25%;
the default 20% endpoint here follows the authors' repository quick-start.
Range extension is explicit, not automatic.

## Paper comparison

`paper_reference.py` extracts the actual vector polylines from Figure 37's
fourth row, rather than estimating values from a screenshot. It also extracts
the optimized rest/deformed shapes and the undeformed specimen photograph.
It requires Poppler (`pdftocairo`), NumPy, and Matplotlib.

```bash
~/miniconda3/envs/sc/bin/python examples/shock_protection/paper_reference.py \
  'additional_info/Huang et al. - 2024 - Optimized shock-protecting microstructures.pdf' \
  build/diamond_reference
```

The periodic reference forces at 10%, 15%, 25%, 40%, and 55% compression are
172.55, 178.93, 177.48, 177.85, and 179.30 N. The experimental curve is separate:
it rises considerably earlier than the periodic curve near 55–60% compression.

Use the stated E = 10⁶ Pa and ν = 0.3. Force is homogenized nominal stress
times the specimen's reference width and thickness. The paper states height
10 cm and thickness 2.6 cm, but does not give this specimen's width numerically.
The undeformed photograph suggests width about 12 cm (approximately 325 pixels
wide versus 270 pixels tall). With this **photograph-derived approximation**,
the reference area is 0.00312 m² and the roughly 178 N plateau corresponds to
about 57 kPa. This is an estimated reconstruction, not a supplied paper input.
The force scale must be fixed before optimization, not fitted afterward.

```bash
~/miniconda3/envs/sc/bin/python examples/shock_protection/optimize.py \
  build/diamond_25 --topology 0004 --target 57000 --graph-radius 3 --strains .10 .15 .20 .25
~/miniconda3/envs/sc/bin/python examples/shock_protection/continue_optimization.py \
  build/diamond_25 build/diamond_extended --maximum-strain 55
```

The continuation helper requires meshio and NumPy. It checks the last accepted
design's stress error, shear, and horizontal strain against 5% before extending
the range by five percentage points. It refuses to extend a failed stage,
including an iteration-limited run that has not reached the criterion.

`--graph-radius 3` increases the inflator's SDF graph neighborhood from the
default two edges to three. The inflator's own troubleshooting documentation
recommends increasing this when periodic-boundary matching fails. Nine failed
diamond trial meshes regenerated successfully with radius three; this avoids
mistaking those numerical meshing failures for invalid shape steps.

Evaluate a saved design without further optimization using `--evaluate-only`
and `--parameters`. Include held-out strains between optimization samples and
cover the full reference plateau. Then run:

```bash
~/miniconda3/envs/sc/bin/python examples/shock_protection/compare_paper.py \
  build/diamond_evaluation build/diamond_reference/reference.json \
  --reference-area .00312
```

The comparison retains raw stresses, reference forces, and the fixed physical
conversion. Its force criterion is RMSE ≤ 10% and maximum relative error ≤ 20%
over the sampled 15–55% plateau, with coverage of that full interval. These are
reproduction tolerances, not the paper's optimization stopping criterion. The
coverage check requires every five-percentage-point sample, not just endpoints.
The separately normalized plot checks curve shape only; it cannot certify amplitude.
Visual inspection of the optimized rest and compressed shapes is still required.
No completed paper reproduction is claimed by these scripts alone.

To inspect an intermediate accepted design, pass `--iteration N` to
`compare_paper.py`. This produces an explicitly labelled in-sample comparison
that cannot pass the independent-validation criterion. Render its geometry with:

```bash
~/miniconda3/envs/sc/bin/python examples/shock_protection/compare_geometry.py \
  build/diamond_25 build/diamond_reference --iteration 20
```

The geometry comparison uses uniform scaling by rest height, preserving aspect
ratio. It does not stretch the candidate to match the paper.

### Measured comparison, 2026-10-05

After correcting the dropped Dirichlet pin described below, both diamond
parameterizations `0004` and `0019` were optimized from their default geometry
at 10%, 15%, 20%, and 25% compression with the estimated 57 kPa target.
`0019` has additional beam control vertices and 55 parameters; the paper does
not identify which diamond parameterization generated Figure 37.
The runs were manually stopped after accepted iterations 34 and 39, respectively,
as contact solves became expensive. Both fail the 25% stage acceptance check;
their checkpoints and `acceptance.json` reports are retained. They did not reach
the configured 100-iteration budget or numerical convergence.

An independent evaluation of `0019` at accepted iteration 38 used the saved
parameters with no further optimization, adding 12.5%, 17.5%, and 22.5% samples.
With E = 10⁶ Pa, ν = 0.3, and the preselected area 0.00312 m²:

| Compression | Simulated stress (Pa) | Simulated force (N) | Paper periodic force (N) |
| --- | --- | --- | --- |
| 10% | 6720.05 | 20.97 | 172.55 |
| 15% | 9884.75 | 30.84 | 178.93 |
| 20% | 11861.82 | 37.01 | 178.19 |
| 25% | 13937.56 | 43.49 | 177.48 |

At the four common strains, the independently computed stresses differ from
the saved optimization samples by at most 0.000225 Pa.
The additional samples reproduce the saved response but do not establish a
plateau: relative force RMSE over the five samples at 15–25% is 79.38%.
The 30–55% range is not validated. Visual inspection also fails: the candidate
has thick, nearly symmetric links rather than the paper's asymmetric thick
links joined by thin hinges. The `0004` iteration-32 geometry is also visibly
different, including a much larger aspect ratio. These are unconverged
intermediate designs, not evidence that the topology cannot reach the target.

Local evidence is retained in `build/shock_0019_iteration38_evaluation/`
(`opt.json`, `optimization.log`, `paper-comparison.json`, `paper-comparison.png`),
`build/shock_0019_iteration38.json`, and the `paper-geometry.png` files in
`build/shock_0019_paper_pinned25/` and `build/shock_0004_paper_pinned25/`.
No modulus adjustment or fitted force multiplier was used.

## Forward incremental load

Requires a Release PolyFEM build with CHOLMOD and the separately built inflator.
Build its mesh-tiling executable if needed:

```bash
cmake --build additional_info/microstructure_inflators/build/release --target isosurface_cli tile
bash examples/shock_protection/run.sh
```

The script creates `build/shock_protection`; it refuses to overwrite an existing
run. Pass a new output directory as its sole argument. `POLYFEM_BIN` and
`INFLATOR_ROOT` override the default binary and external inflator repository.
No dependencies are downloaded or built by the script.

Open `build/shock_protection/output/compression.pvd` in ParaView. The five frames
cover 0%, 5%, 10%, 15%, and 20% compression. Mesh-generation logs, the
single-cell mesh, stitched tiled mesh, single-cell shape velocities, simulation
JSON, and solver log remain in the run directory.

## Model

- A stitched 2×2 **simulation** tile, as used in paper Section 3.5 to allow
  deformation patterns spanning multiple cells. Periodic contact separately
  tiles the simulation boundary for collision detection.
- The inflator produces a cell in [−1, 1]². Scaling the tiled mesh by 0.5 gives
  unit cell side length 1 and simulation tile side length 2.
- Neo-Hookean material: E = 10⁶ Pa, ν = 0.3; quadratic displacement elements,
  linear geometry, quadrature order 5.
- Gyy = −t, with Gxx and the symmetric shear component solved freely.
  Here `time` indexes incremental load, not physical dynamics. Each increment
  starts from the previous solution; inertia is omitted.
- `pin.txt` fixes the fluctuation at input vertex 0, removing translation
  freedom without fixing macroscopic expansion or shear. This is the same
  gauge strategy as the fork, which chooses an interior node instead.
- Meshing resolution, contact distance, and barrier stiffness follow the fork's
  `scripts/optimize.py` defaults. The nonlinear driver and stopping criteria
  are those of this branch, not a reproduction of the fork's solver changes.

The 20% endpoint matches the authors' repository quick-start, with its optional
2×2 tiling enabled. The paper starts optimization at 25% and extends that range
after optimizing the geometry; this forward example does not reproduce that
procedure.

## Validation and limitations

The regular test `differentiable homogenization replaces mesh in place` checks
replacement with a different vertex count, subsequent solves, and continuation.
It exercises both zero-mean and Dirichlet gauges and checks that rebuilding the
solver retains the RHS assembler and boundary constraints. Clearing that assembler
previously dropped the pin and left two translation null modes in the Hessian.
`homogenization initial guess dependencies` covers ordering and invalid links.
The external-inflator test is opt-in:

```bash
python3 examples/shock_protection/optimize.py /tmp/shock-test-input \
  --tiles 1 --strains .01 .02 --grid 24 --max-area .01 --prepare-only
POLYFEM_SHOCK_OPT=/tmp/shock-test-input/opt.json \
  build/release/tests/unit_tests 'inflated periodic shape lifecycle and width gradient'
```

It accepts the `0105`, `0004`, and `0019` parameterizations and checks the width derivative against regenerated-mesh finite differences,
the radius-direction derivative against fixed-connectivity perturbations using
the imported normal velocities, changed mesh sizes, cache identity, and return
to an earlier design. The latter gradient check validates the FE chain rule;
it does not differentiate the discrete remeshing algorithm. Normal velocities
are continuum shape derivatives, and independently remeshed finite differences
can contain discretization noise.
It also checks positive definiteness of the unprojected equilibrium Hessians
at the supplied initial design; this does not certify every optimization trial.

The homogenization driver applies every penalty-weight increase and requires
the configured squared macro-strain error tolerance before switching to exact
constraints. This example uses tolerance 10⁻⁸ and diagonal Newton regularization
without elementwise PSD projection, following the fork's configuration.
Backtracking switches to gradient reduction near equilibrium, using the fork's
`use_grad_norm_tol = 0.001`, with explicit Euclidean norms as in the fork.
The absolute gradient tolerance is `1e-6`; a relative gradient tolerance of
`1e-8` also permits termination when contact roundoff prevents further
absolute-residual reduction. This relative criterion differs from the fork.
The fork's initial-weight direction check is implemented: before an AL solve,
the penalty increases, up to the configured maximum, if the macro gradient
would move a prescribed component away from its target. The paper's
saddle-point perturbation/restart procedure is **not implemented**; the fork's
checked-in driver only reports a failed positive-definiteness check when using
Pardiso. The current branch's nonlinear solver is retained.
The workflow therefore does not certify stable post-buckling equilibria,
mesh convergence, a converged flat stress-strain curve, or reproduction of the
paper's reported optimized designs.

Release validation after the Dirichlet fix passed 248 assertions across six
focused homogenization tests and 47 assertions in the external-inflator test
with `0004`, a 2×2 tile, grid 32, and strains 10–25%. Radius-direction and width
gradient differences from finite differences were 5.28×10⁻⁶ and 6.36×10⁻⁷,
respectively. A contact-bearing grid-64 design also passed the radius-direction
check (difference 1.46×10⁻⁵), but its subsequent width check exceeded the
15-minute test budget; that test did not complete. The broader macro-strain
suite exceeded a 10-minute budget in the 3D forward regression after eight
tests passed. Neither timed-out run is counted as a passing suite.

Earlier smoke-run measurements predated the dropped-pin fix and are not used
as validation of the corrected implementation.

### Additional diagnostics, 2026-10-06

The contact collision-set cache is now owned by each `BarrierContactForm`.
Previously its function-local static position cache was shared across forms:
initializing two forms at identical positions could leave the second form's
collision set empty. The new two-form regression fails before the fix and passes
after it; it and the existing barrier derivative test pass 166 assertions.
Seven focused homogenization tests pass 231 assertions after this fix.

The external-inflator test also passes for `0019` at 1% and 2% compression,
with a 2×2 tile, grid 32, and maximum triangle area 0.003: 29 assertions,
including positive-definite equilibrium Hessians, mesh replacement, and
derivative checks. The radius-direction derivative differs from fixed-mesh
finite differences by 8.34×10⁻⁹; the width derivative differs from remeshed
finite differences by 4.25×10⁻⁹. This does not validate contact-bearing
optimization. A separate test starting from the saved high-compression design
fails in the forward line search before reaching its derivative assertions.

Some solver-setting comparisons remain experimental. The fork's gradient tolerances
use the Euclidean norm, whereas this branch defaults to a mass-weighted L2 norm
with characteristic-force rescaling. Copying the same numeric tolerance does
not preserve either the convergence threshold or the point at which
backtracking switches from energy decrease to gradient-norm decrease.
For one trial, the latter switch stalled with an indefinite Hessian at
Euclidean gradient norm 0.0602. Explicit Euclidean-norm experiments avoid that
particular threshold mismatch, but tight absolute tolerances can still stall
at floating-point noise in contact solves: one run repeated steps below
10⁻¹² with residual 4.05×10⁻⁶ and relative residual 2.33×10⁻⁹.
The example now sets the norm explicitly and accepts relative residual 10⁻⁸.
Experiments with a higher AL cap and a smaller regularization multiplier are
not defaults. No completed optimization or paper agreement is inferred from
these experiments.

The exported stress fields are local stresses. Paper Equation (3) requires
integrating first Piola stress over the material and dividing by the **whole
reference tile area**, including voids (4 for the forward case, 4 × width for
the default optimization); a nodal or material-area average is not the paper's
homogenized stress.
