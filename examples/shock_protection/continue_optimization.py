import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import meshio
import numpy as np


def macro_gradient(mesh):
    points = mesh.points[:, :2]
    displacement = mesh.point_data["displacement"][:, :2]
    _, indices = np.unique(np.round(points, 10), axis=0, return_index=True)
    points, displacement = points[indices], displacement[indices]
    gradient = np.zeros((2, 2))
    for axis in range(2):
        lower = np.flatnonzero(np.isclose(points[:, axis], points[:, axis].min(), atol=1e-8, rtol=0))
        upper = np.flatnonzero(np.isclose(points[:, axis], points[:, axis].max(), atol=1e-8, rtol=0))
        lower = lower[np.argsort(points[lower, 1 - axis])]
        upper = upper[np.argsort(points[upper, 1 - axis])]
        if len(lower) != len(upper) or not np.allclose(points[lower, 1 - axis], points[upper, 1 - axis], atol=1e-8):
            raise RuntimeError("Periodic boundary samples do not match.")
        difference = (displacement[upper] - displacement[lower]) / np.ptp(points[:, axis])
        gradient[:, axis] = difference.mean(axis=0)
        if np.max(np.abs(difference - gradient[:, axis])) > 1e-8:
            raise RuntimeError("Nonperiodic exported displacement.")
    return gradient


def check_stage(directory):
    config = json.loads((directory / "opt.json").read_text())
    ratios, accepted = {}, None
    for line in (directory / "optimization.log").read_text().splitlines():
        match = re.search(r"\[stress_ratio_([\d.]+)\] ([\d.e+-]+)", line)
        if match:
            ratios[float(match[1])] = float(match[2])
        match = re.search(r"Saving iteration (\d+)$", line)
        if match:
            accepted = int(match[1]), dict(ratios)
    if accepted is None:
        raise RuntimeError("No accepted design saved.")
    iteration, ratios = accepted
    records = []
    for state, entry in enumerate(config["states"]):
        state_config = json.loads((directory / entry["path"]).read_text())
        strain = -state_config["constraints"]["macro_displacement_gradient"]["value"][1][1]
        mesh = meshio.read(directory / f"state-{state}/opt_state_{state}_iter_{iteration}.vtu")
        gradient = macro_gradient(mesh)
        error = abs(ratios[strain] - 1)
        records.append(dict(strain=strain, stress_ratio=ratios[strain], stress_relative_error=error,
                            G=gradient.tolist(), passed=bool(error < .05 and abs(gradient[0, 0]) < .05 and abs(gradient[0, 1]) < .05)))
    report = dict(iteration=iteration, passed=all(record["passed"] for record in records), states=records)
    (directory / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Extend an optimized strain range only after the paper's 5% sample criterion passes.")
    parser.add_argument("previous", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--maximum-strain", type=int, default=55, help="Final compression percentage, in increments of five.")
    parser.add_argument("--iterations", type=int, default=100)
    options = parser.parse_args()
    previous = options.previous.resolve()
    config = json.loads((previous / "opt.json").read_text())
    inflation = config["variable_to_simulation"][0]["inflation"]
    target = -1 / config["functionals"][0]["objective"]["objective"][0]["weight"]
    options.directory.mkdir(parents=True, exist_ok=False)
    current = max(round(-json.loads((previous / entry["path"]).read_text())["constraints"]["macro_displacement_gradient"]["value"][1][1] * 100) for entry in config["states"])
    if current % 5 or options.maximum_strain % 5 or options.maximum_strain < current:
        parser.error("Strain endpoints must increase in five-percentage-point increments.")
    while True:
        report = check_stage(previous)
        print(f"{previous}: {json.dumps(report)}", flush=True)
        if not report["passed"]:
            raise SystemExit("Paper stopping criterion not met; refusing to extend the strain range.")
        if current == options.maximum_strain:
            break
        current += 5
        destination = options.directory.resolve() / f"strain-{current}"
        forward = json.loads((previous / config["states"][0]["path"]).read_text())
        extra = ["--energy-line-search"] if forward["solver"]["nonlinear"]["line_search"]["use_grad_norm_tol"] < 0 else []
        lower, upper = config["solver"]["nonlinear"]["box_constraints"]["bounds"]
        radius_upper = max(high for low, high in zip(lower, upper) if low == 0.02)
        subprocess.run([sys.executable, str(Path(__file__).with_name("optimize.py")), str(destination),
                        "--wire", inflation["wire"], "--target", str(target), "--tiles", str(inflation["tiles"]),
                        "--grid", str(inflation["meshing"]["marchingSquaresGridSize"]),
                        "--graph-radius", str(inflation.get("graph_radius", 2)),
                        "--max-area", str(inflation["meshing"]["maxArea"]),
                        "--radius-upper-bound", str(radius_upper), *extra,
                        "--forward-grad-tol", str(forward["solver"]["nonlinear"]["grad_norm_tol"]),
                        "--forward-line-search", forward["solver"]["nonlinear"]["line_search"]["method"],
                        "--iterations", str(options.iterations), "--parameters", str(previous / "parameters.txt"),
                        "--strains", *[str(value / 100) for value in range(10, current + 1, 5)]], check=True)
        previous = destination


if __name__ == "__main__":
    main()
