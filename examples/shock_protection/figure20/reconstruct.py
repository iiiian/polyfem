import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.optimize import minimize


def main():
    parser = argparse.ArgumentParser(description="Reconstruct a Figure 20 rest image, without using stress data.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--design", choices=["A", "B"], default="A")
    options = parser.parse_args()
    directory = options.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    case = Path(__file__).resolve().parent
    root = case.parents[2]
    inflator = root / "additional_info/microstructure_inflators/build/release/isosurface_inflator/isosurface_cli"
    source = next(options.reference.glob(f"{options.design}-rest.*"))
    picture = np.array(Image.open(source).convert("RGB"))
    mask = binary_dilation(picture[:, :, 2].astype(float) - picture[:, :, 0] > 20)
    rows, columns = np.nonzero(mask)
    mask = mask[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    aspect = (mask.shape[1] - 1) / (mask.shape[0] - 1)
    resolution = 128
    target = np.array(Image.fromarray(mask).resize((2 * resolution, 2 * resolution), Image.Resampling.NEAREST))
    target_smooth = gaussian_filter(target.astype(float), 0.6)
    command = [str(inflator), "2D_doubly_periodic", str(case / "extended_chi.obj"),
               "--inflation_graph_radius", "3", "--rasterResolution", f"{resolution}x{resolution}",
               "-r", str(directory / "raster.msh")]

    def expand(shape):
        stem, center_radius, diagonal_radius, corner_radius, horizontal_radius, inset = shape
        diagonal_height = stem + (1 - stem) * inset
        positions = [0.5, (1 + stem) / 2, 0.5, (1 - stem) / 2,
                     (1 + inset) / 2, (1 + diagonal_height) / 2,
                     (1 - inset) / 2, (1 + diagonal_height) / 2,
                     (1 + inset) / 2, (1 - diagonal_height) / 2,
                     (1 - inset) / 2, (1 - diagonal_height) / 2, 0.5]
        return positions + [center_radius] * 2 + [corner_radius] + [diagonal_radius] * 4 + [horizontal_radius] + [0.001] * 8

    def raster(shape):
        subprocess.run(command + ["--params", " ".join(map(str, expand(shape)))],
                       capture_output=True, text=True, check=True)
        with contextlib.redirect_stdout(io.StringIO()):
            mesh = meshio.read(directory / "raster.msh")
        indicator = np.concatenate(mesh.cell_data["indicator"]).reshape(resolution, resolution)
        return np.tile(np.flipud(indicator), (2, 2))

    def objective(shape):
        candidate = gaussian_filter(raster(shape), 0.6)
        return np.mean((candidate - target_smooth) ** 2)

    initial = [0.14, 0.083, 0.077, 0.14, 0.14, 0.8]
    bounds = [(0.07, 0.22), (0.05, 0.11), (0.05, 0.11), (0.08, 0.16), (0.10, 0.17), (0.65, 0.95)]
    if options.design == "B":
        initial = [0.01, 0.04, 0.04, 0.08, 0.025, 0.8]
        bounds = [(0.002, 0.16), (0.02, 0.08), (0.02, 0.08), (0.03, 0.15), (0.02, 0.10), (0.65, 0.95)]
    result = minimize(objective, initial, method="Powell",
                      bounds=bounds,
                      options={"maxfev": 1200, "xtol": 0.0002, "ftol": 1e-6})
    candidate = raster(result.x)
    overlap = np.logical_and(candidate > 0.5, target).sum() / np.logical_or(candidate > 0.5, target).sum()
    parameters = [aspect] + expand(result.x)
    report = {"purpose": "Rest-image reconstruction, not stress fitting or mechanical optimization",
              "source": str(source), "aspect": aspect, "design": options.design,
              "shape_parameters": result.x.tolist(), "intersection_over_union": float(overlap),
              "evaluations": result.nfev, "converged": bool(result.success), "parameters": parameters,
              "image_mask": "Blue interior dilated one pixel to include half of the drawn black outline"}
    (directory / "reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    figure, axes = plt.subplots(1, 3, figsize=(10, 5))
    for axis, data, title in zip(axes, [target, candidate, candidate - target],
                                  [f"Paper {options.design}", "Inflated reconstruction", "Reconstruction minus paper"]):
        axis.imshow(data, extent=(0, aspect, 0, 1), cmap="coolwarm" if "minus" in title else "Blues")
        axis.set_title(title)
        axis.set_axis_off()
    figure.suptitle(f"Rest-shape overlap {overlap:.1%}; measured aspect {aspect:.4f}")
    figure.tight_layout()
    figure.savefig(directory / "reconstruction.png", dpi=180)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
