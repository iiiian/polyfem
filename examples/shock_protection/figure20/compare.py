import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import meshio
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continue_optimization import macro_gradient


def integrate_stress(mesh, young, poisson):
    points = mesh.points[:, :2]
    displacement = mesh.point_data["displacement"][:, :2]
    shear = young / (2 * (1 + poisson))
    lame = young * poisson / (1 - poisson ** 2)
    quadrature = [(1 / 3, 1 / 3, 0.225)]
    for coordinate, weight in [(0.470142064105115, 0.132394152788506), (0.101286507323456, 0.125939180544827)]:
        other = 1 - 2 * coordinate
        quadrature.extend([(coordinate, coordinate, weight), (coordinate, other, weight), (other, coordinate, weight)])
    integral = 0.0
    minimum_determinant = float("inf")
    for block in mesh.cells:
        if block.type != "VTK_LAGRANGE_TRIANGLE" or block.data.shape[1] != 6:
            raise ValueError("Expected quadratic triangle output.")
        vertices = points[block.data]
        for midpoint, endpoints in [(3, [0, 1]), (4, [1, 2]), (5, [2, 0])]:
            if not np.allclose(vertices[:, midpoint], vertices[:, endpoints].mean(axis=1), atol=1e-10):
                raise ValueError("Expected affine triangle geometry.")
        jacobian = np.stack([vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]], axis=2)
        inverse = np.linalg.inv(jacobian)
        area = np.abs(np.linalg.det(jacobian)) / 2
        values = displacement[block.data]
        for first, second, weight in quadrature:
            remaining = 1 - first - second
            derivatives = np.array([[1 - 4 * remaining, 1 - 4 * remaining],
                                    [4 * first - 1, 0], [0, 4 * second - 1],
                                    [4 * (remaining - first), -4 * first],
                                    [4 * second, 4 * first], [-4 * second, 4 * (remaining - second)]])
            gradients = np.einsum("ni,eij->enj", derivatives, inverse)
            deformation = np.eye(2) + np.einsum("eni,enj->eij", values, gradients)
            determinant = np.linalg.det(deformation)
            minimum_determinant = min(minimum_determinant, float(determinant.min()))
            if np.any(determinant <= 0):
                raise ValueError("Inverted deformation at a quadrature point.")
            inverse_transpose = np.linalg.inv(deformation).transpose(0, 2, 1)
            stress = shear * (deformation - inverse_transpose) + lame * np.log(determinant)[:, None, None] * inverse_transpose
            integral += np.sum(area * weight * stress[:, 1, 1])
    return -integral / np.prod(np.ptp(points, axis=0)), minimum_determinant


def main():
    parser = argparse.ArgumentParser(description="Compare independently integrated first-Piola stresses against Figure 20.")
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--design", choices=["A", "B"], required=True)
    options = parser.parse_args()
    config = json.loads((options.run / "forward.json").read_text())
    material = config["materials"]
    reference = json.loads((options.reference / "reference.json").read_text())
    paper = np.array(reference["curves"][options.design])
    records = []
    meshes = {}
    for path in sorted((options.run / "output").glob("step_*.vtu"), key=lambda entry: int(entry.stem.split("_")[-1])):
        step = int(path.stem.split("_")[-1])
        strain = step * config["time"]["dt"]
        mesh = meshio.read(path)
        stress, determinant = integrate_stress(mesh, material["E"], material["nu"])
        record = {"compression_percent": strain * 100, "stress_kPa": stress / 1000,
                  "minimum_det_F": determinant, "G": macro_gradient(mesh).tolist()}
        if not np.isclose(record["G"][1][1], -strain, atol=1e-8, rtol=0):
            raise ValueError("Exported displacement does not satisfy the prescribed macro strain.")
        if paper[0, 0] - 0.01 <= strain * 100 <= paper[-1, 0] + 0.01:
            record["paper_stress_kPa"] = float(np.interp(strain * 100, *paper.T))
        records.append(record)
        if step == 0 or np.isclose(strain, 0.5):
            meshes[round(strain * 100)] = mesh
    if not records:
        parser.error("No forward frames available.")
    compared = [record for record in records if record["compression_percent"] >= 10 and "paper_stress_kPa" in record]
    relative = np.array([record["stress_kPa"] / record["paper_stress_kPa"] - 1 for record in compared])
    report = {"design": options.design, "stress_definition": "Minus material integral of Pyy divided by whole reference tile area",
              "integration": "Quadratic displacement gradients, 7-point degree-5 triangle quadrature; 2D Lame conversion as in PolyFEM",
              "material": material, "completed_steps": len(records) - 1, "expected_steps": config["time"]["time_steps"],
              "records": records, "relative_rmse_from_10_percent": float(np.sqrt(np.mean(relative ** 2))) if relative.size else None,
              "maximum_relative_error_from_10_percent": float(np.max(np.abs(relative))) if relative.size else None}
    (options.run / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    figure, axis = plt.subplots(figsize=(7, 4))
    for name in ("A", "B"):
        axis.plot(*np.array(reference["curves"][name]).T, label=f"Paper {name}")
    axis.plot([record["compression_percent"] for record in records], [record["stress_kPa"] for record in records],
              "o-", label=f"This branch {options.design}")
    axis.set(xlabel="Compression (%)", ylabel="Stress (kPa)", xlim=(0, 65), title="Figure 20: no fitted stress multiplier")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(options.run / "comparison.png", dpi=180)
    if 0 in meshes and 50 in meshes:
        figure, axes = plt.subplots(2, 2, figsize=(8, 7))
        widths = []
        reference_height = None
        simulation_height = np.ptp(meshes[0].points[:, 1])
        for row, (strain, name) in enumerate([(0, "rest"), (50, "compressed50")]):
            image_path = next(options.reference.glob(f"{options.design}-{name}.*"))
            picture = np.array(Image.open(image_path).convert("RGB"))
            mask = binary_dilation(picture[:, :, 2].astype(float) - picture[:, :, 0] > 20)
            rows, columns = np.nonzero(mask)
            picture = picture[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
            scale = reference["image_pdf_scale"][f"{options.design}-{name}"]
            if reference_height is None:
                reference_height = picture.shape[0] * scale[1]
            width = picture.shape[1] * scale[0] / reference_height
            height = picture.shape[0] * scale[1] / reference_height
            axes[row, 0].imshow(picture, extent=(0, width, 0, height))
            widths.append(width)
            axes[row, 0].set_title(f"Paper {options.design}: {strain}%")
            mesh = meshes[strain]
            vertices = mesh.points[:, :2] + mesh.point_data["displacement"][:, :2]
            vertices = (vertices - vertices.min(axis=0)) / simulation_height
            widths.append(np.ptp(vertices[:, 0]))
            for block in mesh.cells:
                axes[row, 1].add_collection(PolyCollection(vertices[block.data[:, [0, 3, 1, 4, 2, 5]]],
                                                         facecolor="#80aaff", edgecolor="none", antialiased=False))
            axes[row, 1].autoscale_view()
            axes[row, 1].set_aspect("equal")
            axes[row, 1].set_title(f"This branch {options.design}: {strain}%")
        for axis in axes.flat:
            axis.set_axis_off()
            axis.set(xlim=(-0.03, max(widths) + 0.03), ylim=(-0.03, 1.03), aspect="equal")
        figure.suptitle("Same rest-height scale; PDF image placement preserved")
        figure.tight_layout()
        figure.savefig(options.run / "geometry.png", dpi=180)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
