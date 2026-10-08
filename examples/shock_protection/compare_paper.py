import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare a fixed-design evaluation with Figure 37's extracted force curve.")
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--iteration", type=int, help="Inspect one accepted optimization iteration; does not count as independent validation.")
    parser.add_argument("--reference-area", type=float, help="Physical specimen reference width times thickness, in square metres.")
    parser.add_argument("--modulus-ratio", type=float, default=1, help="Physical Young's modulus divided by the simulated modulus.")
    options = parser.parse_args()
    if options.modulus_ratio <= 0 or (options.reference_area is not None and options.reference_area <= 0):
        parser.error("Area and modulus ratio must be positive.")
    config = json.loads((options.evaluation / "opt.json").read_text())
    log = (options.evaluation / "optimization.log").read_text()
    independent = config.get("compute_objective", False)
    if independent:
        if options.iteration is not None:
            parser.error("A fixed-design evaluation has no optimization iteration.")
        if "Objective is " not in log:
            parser.error("Fixed-design evaluation did not finish successfully.")
        matches = re.findall(r"\[stress_ratio_([\d.]+)\] ([\d.e+-]+)", log)
    else:
        if options.iteration is None:
            parser.error("Use --evaluate-only output, or explicitly select an accepted --iteration for an in-sample comparison.")
        values, matches = {}, None
        for line in log.splitlines():
            match = re.search(r"\[stress_ratio_([\d.]+)\] ([\d.e+-]+)", line)
            if match:
                values[match[1]] = match[2]
            if re.search(rf"Saving iteration {options.iteration}$", line):
                matches = sorted(values.items(), key=lambda item: float(item[0]))
                break
        if matches is None:
            parser.error("The requested optimization iteration was not saved.")
    if len(matches) != len(config["states"]):
        parser.error("Expected one stress result per evaluation state.")
    stress_weight = config["functionals"][0]["objective"]["objective"][0]["weight"]
    target = -1 / stress_weight
    strains = np.array([float(strain) * 100 for strain, _ in matches])
    stresses = np.array([float(ratio) * target for _, ratio in matches])
    reference = json.loads(options.reference.read_text())
    paper = np.array(reference["curves"]["periodic_simulation"])
    if strains.min() < paper[:, 0].min() or strains.max() > paper[:, 0].max():
        parser.error("Evaluation strains extend beyond the extracted reference curve.")
    paper_force = np.interp(strains, paper[:, 0], paper[:, 1])
    plateau = (strains >= 15) & (strains <= 55)
    if plateau.sum() < 2:
        parser.error("Need multiple samples in the paper's 15–55% plateau.")
    normalized_simulation = stresses / stresses[plateau].mean()
    normalized_paper = paper_force / paper_force[plateau].mean()
    shape_error = normalized_simulation - normalized_paper
    report = {
        "independent_evaluation": independent, "accepted_iteration": options.iteration,
        "compression_percent": strains.tolist(), "simulation_stress_Pa": stresses.tolist(),
        "paper_force_N": paper_force.tolist(),
        "normalization": "Each curve divided by its own mean over sampled 15–55% strains; shape only, not absolute-force validation.",
        "normalized_plateau_rmse": float(np.sqrt(np.mean(shape_error[plateau] ** 2))),
        "normalized_plateau_max_error": float(np.max(np.abs(shape_error[plateau]))),
        "absolute_force_verified": False,
        "covers_paper_plateau": all(bool(np.any(np.isclose(strains, sample, atol=1e-6, rtol=0)))
                                   for sample in range(15, 56, 5))
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(strains, stresses, "o-", label="This branch")
    axes[0].set(ylabel="Homogenized stress (Pa)", title="Unscaled simulation output")
    for name, values in reference["curves"].items():
        axes[1].plot(*np.array(values).T, label="Paper " + name.replace("_", " "))
    if options.reference_area is not None:
        forces = stresses * options.reference_area * options.modulus_ratio
        relative_error = (forces - paper_force) / paper_force
        report.update(reference_area_m2=options.reference_area, modulus_ratio=options.modulus_ratio,
                      simulation_force_N=forces.tolist(),
                      force_plateau_relative_rmse=float(np.sqrt(np.mean(relative_error[plateau] ** 2))),
                      force_plateau_max_relative_error=float(np.max(np.abs(relative_error[plateau]))),
                      force_criterion="Independent fixed-design evaluation: RMSE <= 10% and maximum error <= 20% over sampled 15–55% strains, including every five-percentage-point sample",
                      force_criterion_passed=bool(independent and report["covers_paper_plateau"] and np.sqrt(np.mean(relative_error[plateau] ** 2)) <= .1 and np.max(np.abs(relative_error[plateau])) <= .2))
        axes[1].plot(strains, forces, "o-", label="This branch, supplied physical scaling")
    axes[1].set(ylabel="Force (N)", title="Physical force; no fitted scale")
    axes[2].plot(strains, normalized_paper, label="Paper periodic simulation")
    axes[2].plot(strains, normalized_simulation, "o-", label="This branch")
    axes[2].set(ylabel="Response / own plateau mean", title="Shape-only comparison")
    for axis in axes:
        axis.set_xlabel("Compression (%)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    if not independent:
        figure.suptitle(f"Accepted iteration {options.iteration}: optimization samples only, not independent validation")
        figure.subplots_adjust(top=.85)
    name = "paper-comparison" if independent else f"paper-comparison-iter-{options.iteration}"
    figure.savefig(options.evaluation / f"{name}.png", dpi=180)
    (options.evaluation / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
