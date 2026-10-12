import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continue_optimization import check_stage


def main():
    parser = argparse.ArgumentParser(description="Extend Figure 20 optimization only after Appendix B's independent 1% sampling check.")
    parser.add_argument("previous", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--maximum-strain", type=int, default=55)
    parser.add_argument("--start-strain", type=int)
    parser.add_argument("--iterations", type=int, default=100)
    options = parser.parse_args()
    previous = options.previous.resolve()
    directory = options.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    case = Path(__file__).resolve().parent
    maximum = options.start_strain
    while True:
        config = json.loads((previous / "opt.json").read_text())
        optimized_for = max(round(-json.loads((previous / entry["path"]).read_text())["constraints"]["macro_displacement_gradient"]["value"][1][1] * 100)
                            for entry in config["states"])
        if maximum is None:
            maximum = optimized_for
        if maximum % 5 or options.maximum_strain % 5 or maximum > options.maximum_strain or maximum < optimized_for:
            parser.error("Strain endpoints must increase in five-percentage-point increments.")
        sampled = check_stage(previous)
        if not sampled["passed"]:
            raise SystemExit("Optimization samples fail the 5% criterion; refusing to extend.")
        validation = directory / f"validation-{maximum}"
        if validation.exists():
            validation = directory / f"validation-{maximum}-optimized"
        mesh = previous / f"state-0/opt_state_0_iter_{sampled['iteration']}.obj"
        subprocess.run([sys.executable, str(case / "forward.py"), str(mesh), str(validation),
                        "--increment", "0.01", "--maximum", str(maximum / 100)], check=True)
        subprocess.run([sys.executable, str(case / "compare.py"), str(validation), str(options.reference.resolve()),
                        "--design", "B"], check=True)
        report = json.loads((validation / "comparison.json").read_text())
        target = -1 / config["functionals"][0]["objective"]["objective"][0]["weight"]
        records = [entry for entry in report["records"] if entry["compression_percent"] >= 10 - 1e-8]
        errors = [abs(entry["stress_kPa"] * 1000 / target - 1) for entry in records]
        anchors = [entry for entry in records if np.isclose(entry["compression_percent"] / 5, round(entry["compression_percent"] / 5), atol=1e-8, rtol=0)]
        samples_passed = len(anchors) == (maximum - 10) // 5 + 1 and all(
            abs(entry["stress_kPa"] * 1000 / target - 1) < 0.05
            and abs(entry["G"][0][0]) < 0.05 and abs(entry["G"][0][1]) < 0.05 for entry in anchors)
        complete = report["completed_steps"] == maximum and len(records) == maximum - 9
        passed = complete and max(errors) <= 0.1 and samples_passed
        acceptance = {"optimization": str(previous), "accepted_iteration": sampled["iteration"],
                      "maximum_compression_percent": maximum, "target_stress_Pa": target,
                      "dense_maximum_relative_error": max(errors), "dense_relative_rmse": float(np.sqrt(np.mean(np.array(errors) ** 2))),
                      "complete": complete, "sample_criteria_passed": samples_passed, "passed": passed,
                      "criterion": "Appendix B: every 1% sample in the optimized interval must be within 10% of target"}
        (validation / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
        print(json.dumps(acceptance), flush=True)
        if not passed and maximum <= optimized_for:
            raise SystemExit("Independent dense validation failed; refusing to extend.")
        if passed:
            if maximum == options.maximum_strain:
                (directory / "result.json").write_text(json.dumps(acceptance, indent=2) + "\n")
                break
            maximum += 5
            continue
        inflation = config["variable_to_simulation"][0]["inflation"]
        lower, upper = config["solver"]["nonlinear"]["box_constraints"]["bounds"]
        radius_upper = max(high for low, high in zip(lower, upper) if low == 0.02)
        next_run = directory / f"strain-{maximum}"
        subprocess.run([sys.executable, str(case.parent / "optimize.py"), str(next_run),
                        "--wire", inflation["wire"], "--target", str(target), "--tiles", str(inflation["tiles"]),
                        "--parameters", str(previous / "parameters.txt"), "--iterations", str(options.iterations),
                        "--grid", str(inflation["meshing"]["marchingSquaresGridSize"]),
                        "--max-area", str(inflation["meshing"]["maxArea"]), "--graph-radius", str(inflation["graph_radius"]),
                        "--radius-upper-bound", str(radius_upper), "--forward-line-search", "RobustArmijo", "--forward-grad-tol", "1e-5",
                        "--strains", *[str(strain / 100) for strain in range(10, maximum + 1, 5)]], check=True)
        previous = next_run


if __name__ == "__main__":
    main()
