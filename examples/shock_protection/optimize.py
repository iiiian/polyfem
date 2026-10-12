import argparse
import copy
import json
import os
from pathlib import Path
import subprocess


def write_json(path, value):
    path.write_text(json.dumps(value, indent=4) + "\n")


def main():
    case = Path(__file__).resolve().parent
    root = case.parent.parent
    parser = argparse.ArgumentParser(description="Optimize inflated periodic microstructures using VarForm.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--inflator-root", type=Path, default=root / "additional_info/microstructure_inflators")
    parser.add_argument("--polyfem", type=Path, default=root / "build/release/PolyFEM_bin")
    parser.add_argument("--topology", default="0105", help="Inflator topology ID, e.g. 0004 or 0105 (default).")
    parser.add_argument("--wire", type=Path, help="Custom periodic OBJ edge mesh, instead of a topology ID.")
    parser.add_argument("--target", type=float, default=5000)
    parser.add_argument("--strains", type=float, nargs="+", default=[0.1, 0.15, 0.2])
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--graph-radius", type=int, default=2, help="Inflation graph neighborhood radius in edges.")
    parser.add_argument("--max-area", type=float, default=0.001)
    parser.add_argument("--radius-upper-bound", type=float, default=0.15)
    parser.add_argument("--forward-grad-tol", type=float, default=1e-6)
    parser.add_argument("--forward-line-search", choices=["Backtracking", "Armijo", "RobustArmijo"], default="Backtracking")
    parser.add_argument("--energy-line-search", action="store_true", help="Keep energy-based backtracking near buckling instead of switching to gradient-norm decrease.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the supplied design without changing its parameters.")
    parser.add_argument("--width-only", action="store_true", help="Optimize only aspect ratio before releasing the other shape parameters (paper Section 3.5).")
    options = parser.parse_args()
    if options.target <= 0 or options.tiles < 1 or options.iterations < 1 or options.grid < 2 or options.max_area <= 0 or options.graph_radius < 1 or options.radius_upper_bound < 0.02 or options.forward_grad_tol <= 0:
        parser.error("Target stress, tile count, iterations, and meshing settings must be positive.")
    if any(not 0 < strain < 1 for strain in options.strains) or sorted(set(options.strains)) != options.strains:
        parser.error("Strains must be distinct, increasing compression fractions between zero and one.")
    inflator_root = options.inflator_root.resolve()
    inflator = inflator_root / "build/release/isosurface_inflator/isosurface_cli"
    tiler = inflator.parent / "tile"
    polyfem = options.polyfem.resolve()
    wire = options.wire.resolve() if options.wire else inflator_root / "data/patterns/2D/topologies" / f"{options.topology}.obj"
    for executable in (inflator, tiler, polyfem):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            parser.error(f"Missing executable: {executable}")
    if not wire.is_file():
        parser.error(f"Missing topology: {wire}")
    directory = options.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    meshing = json.loads((case / "meshing.json").read_text())
    meshing.update(maxArea=options.max_area, marchingSquaresGridSize=options.grid)
    write_json(directory / "meshing.json", meshing)
    command = [str(inflator), "2D_doubly_periodic", str(wire), "--defaultThickness", "0.05",
               "--inflation_graph_radius", str(options.graph_radius),
               "-m", "meshing.json", "cell.msh"]
    result = subprocess.run(command, cwd=directory, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=True)
    (directory / "inflation.log").write_text(result.stdout)
    lines = result.stdout.splitlines()
    marker = next(index for index, line in enumerate(lines) if line.startswith("Inflating default parameters:"))
    initial = [1.0] + [float(value) for value in lines[marker + 1].split()]
    positional, thickness, blending = map(int, lines[marker + 2].split())
    if options.parameters:
        text = options.parameters.read_text().strip()
        initial = json.loads(text) if text.startswith("[") else [float(value) for value in text.splitlines()[-1].split(":")[-1].split()]
        if len(initial) != 1 + positional + thickness + blending:
            parser.error("Parameter vector must contain cell width and all inflator parameters.")
        subprocess.run([str(inflator), "2D_doubly_periodic", str(wire), "-m", "meshing.json",
                        "--inflation_graph_radius", str(options.graph_radius),
                        "--params", " ".join(map(str, initial[1:])), "cell.msh"], cwd=directory, check=True)
    subprocess.run([str(tiler), "-t", f"{options.tiles}x{options.tiles}", "cell.msh", "tiled.msh"],
                   cwd=directory, check=True)
    (directory / "pin.txt").write_text("0 0 0\n")
    forward = json.loads((case / "forward.json").read_text())
    forward.pop("time")
    forward["geometry"][0]["transformation"]["scale"] = [0.5 * initial[0], 0.5]
    forward["solver"]["adjoint_linear"] = {"solver": "Eigen::SimplicialLDLT"}
    forward["output"]["advanced"]["save_time_sequence"] = False
    forward["solver"]["nonlinear"]["grad_norm_tol"] = options.forward_grad_tol
    forward["solver"]["nonlinear"]["line_search"]["method"] = options.forward_line_search
    forward["solver"]["augmented_lagrangian"]["nonlinear"]["line_search"]["method"] = options.forward_line_search
    if options.energy_line_search:
        forward["solver"]["nonlinear"]["line_search"]["use_grad_norm_tol"] = -1
        forward["solver"]["augmented_lagrangian"]["nonlinear"]["line_search"]["use_grad_norm_tol"] = -1
    states, functionals, stopping_conditions = [], [], []
    for index, strain in enumerate(options.strains):
        state = copy.deepcopy(forward)
        state["constraints"]["macro_displacement_gradient"]["value"] = [[0, 0], [0, -strain]]
        state["output"]["directory"] = str(directory / f"state-{index}")
        write_json(directory / f"state-{index}.json", state)
        states.append({"path": f"state-{index}.json", "initial_guess": index - 1})
        # StressForm integrates Pyy over the solid; divide by the whole reference tile area, including voids.
        stress_ratio = {
            "type": "divide", "print_energy": f"stress_ratio_{strain:g}",
            "objective": [
                {"type": "stress", "dimensions": [1, 1], "state": index, "weight": -1 / options.target},
                {"type": "parametrized_product", "weight": options.tiles ** 2,
                 "parametrization": [{"type": "slice", "from": 0, "to": 1, "last": len(initial)}]}
            ]
        }
        stress_error = {"type": "soft_constraint", "soft_bound": [1, 1], "power": 2, "objective": stress_ratio}
        functionals.append(stress_error)
        # The fork's stopping threshold is a 5% relative error squared: 0.05**2 = 0.0025.
        stopping_conditions.append({"type": "plus-const", "value": -0.0025, "objective": stress_error})
        for dimensions in ([0, 1], [0, 0]):
            penalty = {"type": "power", "power": 2, "weight": 50,
                       "objective": {"type": "homo_disp_grad", "dimensions": dimensions, "state": index}}
            functionals.append(penalty)
            stopping_conditions.append({"type": "plus-const", "value": -0.0025,
                                        "objective": dict(penalty, weight=1)})
    config = {
        "compute_objective": options.evaluate_only,
        "parameters": "auto", "states": states,
        "variable_to_simulation": [{
            "type": "inflated-periodic-shape", "state": list(range(len(states))), "composition": [],
            "inflation": {"wire": str(wire),
                          "work_directory": str(directory / "mesh-generations"), "tiles": options.tiles,
                          "graph_radius": options.graph_radius,
                          "initial": initial, "meshing": meshing}
        }],
        "functionals": functionals, "stopping_conditions": stopping_conditions,
        "solver": {
            "max_threads": 16,
            "advanced": {"solve_in_parallel": False, "enable_slim": False, "smooth_line_search": False},
            "nonlinear": {
                "solver": "L-BFGS-B", "max_iterations": options.iterations, "grad_norm_tol": 1e-4,
                "allow_out_of_iterations": True,
                "line_search": {"method": "Backtracking", "min_step_size": 0.01},
                "box_constraints": {
                    "bounds": [[0.2] + [0] * positional + [0.02] * thickness + [0.001] * blending,
                               [5] + [1] * positional + [options.radius_upper_bound] * thickness + [0.1] * blending],
                    "max_change": [0.2] + [0.05] * positional + [0.005] * thickness + [0.01] * blending
                }
            }
        },
        "output": {"directory": str(directory / "optimization"), "save_frequency": 1,
                   "solution": str(directory / "parameters.txt"), "log": {"level": "debug"}}
    }
    if options.width_only:
        box = config["solver"]["nonlinear"]["box_constraints"]
        box["bounds"][0][1:] = initial[1:]
        box["bounds"][1][1:] = initial[1:]
        box["max_change"][1:] = [0] * (len(initial) - 1)
    write_json(directory / "opt.json", config)
    print(f"Optimization configuration: {directory / 'opt.json'}", flush=True)
    if not options.prepare_only:
        with (directory / "optimization.log").open("w") as log:
            subprocess.run([str(polyfem), "--ns", "-j", str(directory / "opt.json")], stdout=log,
                           stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
