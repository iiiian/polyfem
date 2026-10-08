import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Summarize the independently validated Figure 20 A-to-B comparison.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("directory", type=Path)
    options = parser.parse_args()
    reports = {"A": json.loads((options.baseline / "comparison.json").read_text()),
               "B": json.loads((options.optimized / "comparison.json").read_text())}
    reference = json.loads((options.reference / "reference.json").read_text())
    if reports["A"]["material"] != reports["B"]["material"]:
        raise ValueError("A and B must use the same material.")
    options.directory.mkdir(parents=True, exist_ok=False)
    summary = {"baseline": str(options.baseline.resolve()), "optimized": str(options.optimized.resolve()),
               "reference": str(options.reference.resolve()), "material": reports["A"]["material"],
               "stress_scale_fitted": False, "comparisons": {}}
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for name, color in [("A", "#1769aa"), ("B", "#d45b00")]:
        report = reports[name]
        if report["completed_steps"] != report["expected_steps"]:
            raise ValueError(f"{name}: incomplete forward solve.")
        values = np.array([[record["compression_percent"], record["stress_kPa"]] for record in report["records"]])
        paper = np.array(reference["curves"][name])
        axis.plot(*paper.T, color=color, label=f"Paper {name}")
        axis.plot(*values.T, "o--", markersize=3, color=color, label=f"This branch {name}")
        comparison = {"maximum_compression_percent": float(values[-1, 0]), "intervals": {}}
        for lower, upper in [(1, 9), (10, 50), (10, 55), (10, 60)]:
            if upper > min(values[-1, 0], paper[-1, 0]) + 0.01:
                continue
            selected = values[(values[:, 0] >= lower - 1e-8) & (values[:, 0] <= upper + 1e-8)]
            errors = selected[:, 1] / np.interp(selected[:, 0], *paper.T) - 1
            comparison["intervals"][f"{lower}-{upper}"] = {
                "samples": len(selected), "relative_rmse": float(np.sqrt(np.mean(errors ** 2))),
                "maximum_relative_error": float(np.max(np.abs(errors)))}
        comparison["samples"] = [{"compression_percent": strain,
                                  "stress_kPa": float(np.interp(strain, *values.T)),
                                  "paper_stress_kPa": float(np.interp(strain, *paper.T))}
                                 for strain in [5, 10, 20, 30, 40, 50, 55, 60]
                                 if strain <= min(values[-1, 0], paper[-1, 0]) + 0.01]
        summary["comparisons"][name] = comparison
    axis.set(xlabel="Compression (%)", ylabel="Homogenized stress (kPa)", xlim=(0, 65),
             title="Figure 20 A → B: E = 1 MPa, ν = 0.3; no fitted stress multiplier")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(options.directory / "curves.png", dpi=180)
    (options.directory / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
