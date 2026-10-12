# Figure 20: A to B

The reference is page 13 of the local *Optimized shock-protecting
microstructures* paper. A and B share the extended chi connectivity; C is a
different topology. The ordinate is already homogenized stress in kPa.
No specimen-area estimate, modulus fit, or multiplicative stress fit is used.

**Result:** the independently remeshed optimized design agrees with paper B to
2.39% relative RMS error and 4.75% maximum relative error over 10–60%
compression, including the rise after the plateau. It does **not** recover the
exact published geometry or the initial small-strain stiffness. The strict
5% optimization-sample stopping criterion remains unmet at 55%; the independent
10% dense-sampling criterion passes. See the final comparison below.

`figure20_reference.py` extracts the original embedded geometry images and the
vector stress curves. B's mean stress at 15–50% compression is approximately
1.080 kPa. We use 1080 Pa as the optimization target; this is inferred from the
published curve, not a recovered author input file.

## Initial geometry

`extended_chi.obj` is a reconstruction of A's connectivity, not a supplied
author geometry. The local `periodic_x.obj` has related connectivity but visibly
different aspect ratio and thickness. The enumerated 105 graphs do not identify
the Figure 20 baseline's parameters.

`reconstruct.py` fits six symmetric geometric quantities to A's **rest image
only**, with the aspect ratio measured independently from that image. Stress
data never enters this fit. It includes horizontal ties, a short vertical
central stem, and subdivided diagonal members. Subsequent mechanical shape
optimization releases the full inflator parameterization (30 variables including
width), rather than enforcing those six reconstruction parameters.

The first reconstruction has aspect 0.8564 and 93.2% binary silhouette overlap.
This is approximate geometry; beam-thickness and joint differences remain.
One reconstructed horizontal radius is 0.16265, above the authors' script bound
0.15. The reproduction uses an explicit 0.17 radius upper bound to contain A,
rather than silently clipping the starting geometry.

## Reproduction commands

Run from the repository root. Every runner refuses to overwrite its directory.

```bash
PYTHON=~/miniconda3/envs/sc/bin/python
$PYTHON examples/shock_protection/figure20_reference.py \
  'additional_info/Huang et al. - 2024 - Optimized shock-protecting microstructures.pdf' \
  build/figure20-paper
$PYTHON examples/shock_protection/figure20/reconstruct.py \
  build/figure20-paper build/figure20-A-reconstruction
$PYTHON examples/shock_protection/optimize.py build/figure20-A-evaluation \
  --wire examples/shock_protection/figure20/extended_chi.obj \
  --parameters build/figure20-A-reconstruction/parameters.json \
  --target 1080 --strains .05 --grid 96 --graph-radius 3 \
  --radius-upper-bound .17 --forward-line-search RobustArmijo --forward-grad-tol 1e-5 \
  --evaluate-only
$PYTHON examples/shock_protection/figure20/forward.py \
  build/figure20-A-evaluation/mesh-generations/0/mesh.msh build/figure20-A-validation
$PYTHON examples/shock_protection/figure20/compare.py \
  build/figure20-A-validation build/figure20-paper --design A
```

Only after checking A's response, run the A-to-B optimization at 10%, 15%, 20%,
and 25%, then extend the strain range:

```bash
$PYTHON examples/shock_protection/optimize.py build/figure20-A-to-B \
  --wire examples/shock_protection/figure20/extended_chi.obj \
  --parameters examples/shock_protection/figure20/A-reconstructed.json \
  --target 1080 --strains .1 .15 .2 .25 --grid 64 --max-area .001 \
  --graph-radius 3 --radius-upper-bound .17 \
  --forward-line-search RobustArmijo --forward-grad-tol 1e-5
$PYTHON examples/shock_protection/figure20/continue_validated.py \
  build/figure20-A-to-B build/figure20-continuation build/figure20-paper
```

Independently reinflate the accepted parameters with `--grid 96 --max-area
.0005 --evaluate-only --strains .01`, then use `forward.py --increment .01
--maximum .6` and `compare.py --design B` for the final comparison. `report.py
BASELINE_FORWARD OPTIMIZED_FORWARD REFERENCE DIRECTORY` combines both validated
curves and explicitly reports the low-strain mismatch separately from the
optimized interval. A saved optimizer iteration alone is not independent
validation.

`continue_validated.py PREVIOUS DIRECTORY REFERENCE` first checks the 5%
optimization-sample criterion, independently solves the saved rest mesh from
zero compression in 1% increments, and requires every sample from 10% to the
current endpoint to be within 10% of target. Only then does it extend the
optimization interval by 5%, following Appendix B. Failed solves or checks stop
the sequence. The default final endpoint is 55%. At a new endpoint it tests
the unchanged design first: if the 5% sample criterion and dense validation
already pass, no shape update is needed. Otherwise it optimizes the expanded
interval and validates again. This avoids the nonlinear solver's unconditional
first step before its custom stopping-condition check.

The initial parameters are also frozen in `A-reconstructed.json`; this avoids
depending on repeatability of raster fitting when rerunning the mechanical
experiment. `B-optimized.json` freezes the independently validated checkpoint.
To reproduce its forward result without rerunning optimization:

```bash
$PYTHON examples/shock_protection/optimize.py build/figure20-B-mesh \
  --wire examples/shock_protection/figure20/extended_chi.obj \
  --parameters examples/shock_protection/figure20/B-optimized.json \
  --target 1080 --strains .01 --grid 96 --max-area .0005 \
  --graph-radius 3 --radius-upper-bound .17 \
  --forward-line-search RobustArmijo --forward-grad-tol 1e-5 --evaluate-only
$PYTHON examples/shock_protection/figure20/forward.py \
  build/figure20-B-mesh/mesh-generations/0/mesh.msh build/figure20-B-validation \
  --increment .01 --maximum .6
$PYTHON examples/shock_protection/figure20/compare.py \
  build/figure20-B-validation build/figure20-paper --design B
$PYTHON examples/shock_protection/figure20/report.py \
  build/figure20-A-validation build/figure20-B-validation \
  build/figure20-paper build/figure20-report
```

## Numerical checks

- Material stays Neo-Hookean, E = 10⁶ Pa, nu = 0.3, with quadratic displacement
  elements and a 2×2 periodic simulation tile.
- The original near-equilibrium gradient-norm line search stalls near A's
  first buckling point with an indefinite Hessian. Energy-only backtracking
  solves the identical 5% case in about 11 seconds, with final residual
  3.1e-9. This changes the solver criterion, not geometry or constitutive data.
- At contact, the 1e-6 absolute tolerance stalls at residuals around 2–4e-6,
  displacement steps around 1e-12, and energy changes around 1e-12. The Figure
  20 runner uses absolute tolerance 1e-5 and retains relative tolerance 1e-8.
- Strict energy decrease can also fail at floating-point resolution. The
  existing polysolve `RobustArmijo` method accepts sufficient energy decrease
  and uses directional derivatives when energy subtraction loses accuracy.
  With this method, the reconstructed graph passes all 31 assertions of the
  external-inflator lifecycle/gradient test at 1% and 2% compression: radius
  derivative error 1.24e-7, width derivative error 2.04e-7. Strict backtracking
  failed during the same test's restored-mesh solve. This is a low-strain
  derivative check, not validation of contact-bearing derivatives.
- `compare.py` integrates first Piola stress from exported quadratic
  displacement fields with degree-5 triangle quadrature, divided by the whole
  reference tile area including voids. It uses the same 2D Lame conversion as
  `MatParams.cpp`, not a nodal stress average.
- The integrator passes an analytic affine-deformation check. At 5% strain it
  gives 2986.44336 Pa, matching `StressForm`'s separate static evaluation.
- Polysolve logs its custom objective-stop status at error severity even when
  the sample criteria pass. The CLI treats that status as success; the separate
  sample and dense-validation reports determine acceptance here.

The comparison JSON records how many load steps completed; partial output is
not a completed reproduction. Setup, derivative tests, and baseline checks
alone do not establish agreement with B.

## Completed checks

The baseline run `build/figure20-A-forward-robust` completes 0–50% compression
at 2.5% increments. Over 10–50%, relative RMS error against Figure 20 A is
14.29%, maximum 21.67%. At 10%, 30%, and 50%, simulated stresses are approximately
2.952, 2.814, and 3.888 kPa, versus 2.548, 2.420, and 3.196 kPa in the paper.
The compressed geometry has the same folded-cell pattern, with opposite
handedness of the symmetry-breaking deformation. The baseline is approximate,
not an exact recovery of the authors' geometry.

The unrestricted A-to-B run `build/figure20-A-to-B-robust25` reaches the
optimization stopping criterion at saved iteration 12. Its independent
1%-increment validation in `build/figure20-B-continuation/validation-25` passes:
maximum target error 2.44%; RMS error against the published B curve 0.85%,
maximum 2.12%, over 10–25%. This does not establish agreement at larger strains.

The unchanged design passes dense validation through 40%, but exceeds target
by 14.2% at 45%. The expanded optimization in
`build/figure20-B-validated/strain-45` converges after six updates and passes its
independent check. At 50%, stress rises too early, to 1.597 kPa; the next stage
`strain-50` converges after ten updates. Its independent 1%-increment validation
in `validation-50-optimized` has 1.32% relative RMS error against paper B,
maximum 3.80%, over 10–50%. Maximum target error is 4.11%.

The first 55% check was interrupted by SIGTERM before completion, without a
solver failure report. It is not counted as validation. The continuation
restarts from the accepted 50% design in `build/figure20-B-final`.

The first aspect-ratio-only comparison run was stopped after the unrestricted
run converged. A later complete width-only run,
`build/figure20-A-to-B-width-first`, converges in eight updates to aspect 1.5875,
versus approximately 1.2427 in paper B. Only width changes in this diagnostic;
it is not followed by full shape optimization and is not used as a seed for
the reported result. Thus it does not establish whether the paper's full
warm-start alternative would recover a closer geometry. A separate symmetric
fit of B's rest image has only 76.3% silhouette overlap. A separate 10%
fixed-shape evaluation of that approximation gives 0.791 kPa, versus the
published 1.076 kPa. This diagnostic neither recovers the authors' geometry
nor establishes a normalization discrepancy. It is not used as an optimization
starting point or as evidence that A-to-B optimization succeeded.

## Final comparison

The reported B design is **accepted iteration 14** of
`build/figure20-B-final/strain-55`, reached from reconstructed A through the
25%, 45%, 50%, and 55% optimization stages. The 30%, 35%, and 40% extensions
already passed without further shape changes. Independent 1%-increment
reloading of iteration 14 agrees with the paper over 10–55% to 1.65% RMS and
4.60% maximum relative error on the optimization mesh.

The checkpoint was then reinflated at marching-squares resolution 96 instead
of 64 and maximum triangle area 0.0005 instead of 0.001. The resulting forward
mesh has 14,772 P2 triangles versus 7,616 before refinement. A fresh solve from
zero through **60%**, in 1% increments, completes all 60 steps. Maximum stress
change under this combined geometry/FE refinement is 1.64% over 10–55%.

| Compression | This branch B (kPa) | Paper B (kPa) |
| --- | ---: | ---: |
| 5% | 0.97344 | 1.03073 |
| 10% | 1.02540 | 1.07648 |
| 20% | 1.05680 | 1.08799 |
| 30% | 1.06693 | 1.08406 |
| 40% | 1.06436 | 1.07596 |
| 50% | 1.04968 | 1.06838 |
| 55% | 1.16674 | 1.12979 |
| 60% | 2.14468 | 2.08053 |

- Relative RMS error is 2.44% over 10–55%, and 2.39% over 10–60%; maximum
  relative error is 4.75% in both intervals. These are pointwise relative
  errors against the extracted paper curve, not a fitted normalization.
- At 1–9%, relative RMS error is 26.54%, maximum 54.38%. Matching the plateau
  does not establish agreement with the initial tangent stiffness.
- Every 1% stress sample from 10–55% is within 8.04% of the 1080 Pa target,
  satisfying Appendix B's independent 10% criterion. The optimization's
  stricter 5% sample criterion is **not** satisfied at 55%: iteration 14 is
  9.42% high on the optimization mesh. Further optimization was manually
  stopped after accepted iteration 15, amid large stress changes in trial
  steps. Iteration 15 is not the reported/refined checkpoint; strict optimizer
  convergence is not claimed.
- Visual inspection covers both rest and 50%-compressed shapes. Connectivity
  and the folded-cell pattern agree qualitatively, but the result is narrower
  (aspect 0.95085 versus approximately 1.2427), has different tie thickness and
  diagonal asymmetry, and folds with opposite handedness. Images use the same
  rest-height scale, preserve PDF image placement, and do not stretch or
  reflect the simulated geometry to improve the comparison.
- All exported fields pass the periodic affine-jump and prescribed Gyy checks.
  Minimum deformation determinant at the sampled quadrature points is 0.6453
  through 60%; this is not an exhaustive pointwise inversion certificate.

Artifacts in this working tree:

- `build/figure20-report/curves.png`: both A and B versus the paper.
- `build/figure20-report/results.json`: numeric comparison and provenance.
- `build/figure20-B-iter14-refined-forward/geometry.png`: B rest and 50% shapes.
- `build/figure20-B-iter14-refined-forward/comparison.json`: all 61 frames,
  stresses, macro gradients, and quadrature determinant checks.
- `build/figure20-A-forward-robust/geometry.png`: corresponding A shape check.

This is a close reproduction of B's plateau and densification curve, with
approximate A geometry and a different optimized geometry. It is not an exact
recovery of the authors' Figure 20 configuration. All mechanical solves and
shape updates run in this branch's VarForm implementation; the external
inflator only supplies geometry and shape velocities.
