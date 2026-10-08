import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import meshio
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Inspect optimized geometry against Figure 37 without changing its aspect ratio.")
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--iteration", type=int)
    options = parser.parse_args()
    config = json.loads((options.run / "opt.json").read_text())
    iterations = set.intersection(*[
        {int(path.stem.split("_")[-1]) for path in (options.run / f"state-{state}").glob(f"opt_state_{state}_iter_*.vtu")}
        for state in range(len(config["states"]))
    ])
    if not iterations:
        parser.error("No iteration has exported meshes for every strain state.")
    iteration = options.iteration if options.iteration is not None else max(iterations)
    if iteration not in iterations:
        parser.error("The requested iteration has not exported every strain state.")
    picture = np.array(Image.open(options.reference / "paper-rest.png").convert("RGB"))
    foreground = (picture[:, :, 2].astype(float) - picture[:, :, 0]) > 20
    rows, columns = np.nonzero(foreground)
    picture = picture[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    paper_aspect = picture.shape[1] / picture.shape[0]
    figure, axes = plt.subplots(1, 4, figsize=(16, 5))
    axes[0].imshow(picture, extent=(0, paper_aspect, 0, 1))
    axes[0].set_title("Paper: optimized rest shape")
    for axis, state, deformed in zip(axes[1:], [0, 0, len(config["states"]) - 1], [False, True, True]):
        mesh = meshio.read(options.run / f"state-{state}/opt_state_{state}_iter_{iteration}.vtu")
        points = mesh.points[:, :2].copy()
        height = np.ptp(points[:, 1])
        if deformed:
            points += mesh.point_data["displacement"][:, :2]
        points = (points - points.min(axis=0)) / height
        polygons = []
        for block in mesh.cells:
            if block.type != "VTK_LAGRANGE_TRIANGLE" or block.data.shape[1] != 6:
                raise ValueError("Expected quadratic triangle visualization cells.")
            polygons.extend(points[block.data[:, [0, 3, 1, 4, 2, 5]]])
        axis.add_collection(PolyCollection(polygons, facecolors="#80aaff", edgecolors="none"))
        state_config = json.loads((options.run / config["states"][state]["path"]).read_text())
        strain = -state_config["constraints"]["macro_displacement_gradient"]["value"][1][1]
        axis.set_title(f"Iteration {iteration}: {strain:.0%} compression" if deformed else f"Iteration {iteration}: rest")
        axis.autoscale_view()
    extent = max(axis.get_xlim()[1] for axis in axes)
    for axis in axes:
        axis.set(xlim=(-.05, extent + .05), ylim=(-.05, 1.05), aspect="equal")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Uniform scaling by rest height only; no horizontal stretching or fitted deformation")
    figure.tight_layout()
    figure.savefig(options.run / "paper-geometry.png", dpi=180)


if __name__ == "__main__":
    main()
