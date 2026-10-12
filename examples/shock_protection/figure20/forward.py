import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Independently compress an already generated Figure 20 design.")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--increment", type=float, default=0.025)
    parser.add_argument("--maximum", type=float, default=0.5)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--line-search", choices=["Backtracking", "RobustArmijo"], default="RobustArmijo")
    options = parser.parse_args()
    if options.increment <= 0 or options.grad_tol <= 0 or not 0 < options.maximum < 1:
        parser.error("Increment and gradient tolerance must be positive; maximum compression must be between zero and one.")
    steps = round(options.maximum / options.increment)
    if abs(steps * options.increment - options.maximum) > 1e-10:
        parser.error("Maximum compression must be a positive integer multiple of the increment.")
    directory = options.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    case = Path(__file__).resolve().parent.parent
    config = json.loads((case / "forward.json").read_text())
    config["geometry"][0]["mesh"] = str(options.mesh.resolve())
    config["geometry"][0].pop("transformation")
    config["boundary_conditions"]["dirichlet_boundary"] = [str(directory / "pin.txt")]
    config["time"].update(dt=options.increment, time_steps=steps)
    config["output"]["directory"] = str(directory / "output")
    config["solver"]["nonlinear"]["line_search"]["use_grad_norm_tol"] = -1
    config["solver"]["nonlinear"]["grad_norm_tol"] = options.grad_tol
    config["solver"]["nonlinear"]["line_search"]["method"] = options.line_search
    config["solver"]["augmented_lagrangian"]["nonlinear"]["line_search"]["method"] = options.line_search
    config["solver"]["augmented_lagrangian"]["nonlinear"]["line_search"]["use_grad_norm_tol"] = -1
    (directory / "pin.txt").write_text("0 0 0\n")
    (directory / "forward.json").write_text(json.dumps(config, indent=4) + "\n")
    with (directory / "forward.log").open("w") as log:
        subprocess.run([str(case.parents[1] / "build/release/PolyFEM_bin"), "-j", str(directory / "forward.json")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
