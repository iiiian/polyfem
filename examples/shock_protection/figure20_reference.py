import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Extract Figure 20 curves and original geometry images without force rescaling.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("directory", type=Path)
    options = parser.parse_args()
    options.directory.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temporary:
        svg = Path(temporary) / "page13.svg"
        subprocess.run(["pdftocairo", "-f", "13", "-l", "13", "-svg", str(options.pdf), str(svg)], check=True)
        root = ET.parse(svg).getroot()
    colors = {"rgb(12.205505%, 46.702576%, 70.506287%)": "A",
              "rgb(100%, 49.804688%, 5.499268%)": "B",
              "rgb(17.30957%, 62.693787%, 17.30957%)": "C"}
    images = {"source-49": "A-rest", "source-52": "A-compressed50",
              "source-46": "B-rest", "source-40": "B-compressed50",
              "source-43": "C-rest", "source-37": "C-compressed50"}
    curves = {}
    image_scales = {}
    extracted_images = set()
    for element in root.iter():
        reference = element.get("{http://www.w3.org/1999/xlink}href", "")
        placed_name = images.get(reference.removeprefix("#"))
        if placed_name:
            transform = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", element.get("transform", ""))))
            if len(transform) != 6 or transform[1] != 0 or transform[2] != 0:
                raise ValueError("Expected axis-aligned image placement.")
            image_scales[placed_name] = [transform[0], transform[3]]
        name = images.get(element.get("id"))
        if name:
            data = element.get("{http://www.w3.org/1999/xlink}href", "")
            if not data.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
                raise ValueError("Expected embedded PNG or JPEG geometry images.")
            extension = "png" if data.startswith("data:image/png") else "jpg"
            (options.directory / f"{name}.{extension}").write_bytes(base64.b64decode(data.split(",", 1)[1]))
            extracted_images.add(name)
        path = element.get("d", "")
        color = element.get("stroke", "")
        if color not in colors or not path.startswith("M 3308.945312 5324."):
            continue
        if re.search("[ACHQSTVZachqstvz]", path):
            raise ValueError("Expected polyline data.")
        coordinates = np.array(list(map(float, re.findall(r"-?\d+(?:\.\d+)?", path)))).reshape(-1, 2)
        coordinates[:, 0] = (coordinates[:, 0] - 3308.945312) * 20 / (3868.984375 - 3308.945312)
        coordinates[:, 1] = (coordinates[:, 1] - 5324.648438) * 2 / (5623.4375 - 5324.648438)
        if colors[color] in curves:
            raise ValueError("Ambiguous curve selection.")
        curves[colors[color]] = coordinates.tolist()
    if set(curves) != set(colors.values()) or extracted_images != set(images.values()) or set(image_scales) != extracted_images:
        raise ValueError("Figure 20 does not match this paper version.")
    result = {"source": options.pdf.name, "page": 13, "figure": 20,
              "method": "PDF vector polylines calibrated against labeled ticks; no fitted stress scaling",
              "columns": ["compression_percent", "stress_kPa"],
              "axis_calibration_svg": {"x_0": 3308.945312, "x_20": 3868.984375,
                                       "y_0": 5324.648438, "y_2": 5623.4375},
              "curves": curves, "image_pdf_scale": image_scales}
    (options.directory / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
    figure, axis = plt.subplots(figsize=(7, 4))
    for name, values in curves.items():
        coordinates = np.array(values)
        np.savetxt(options.directory / f"{name}.csv", coordinates, delimiter=",",
                   header="compression_percent,stress_kPa", comments="")
        axis.plot(*coordinates.T, label=name)
    axis.set(xlabel="Compression (%)", ylabel="Stress (kPa)", xlim=(0, 80), ylim=(0, 4.5),
             title="Paper Figure 20 — extracted reference")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(options.directory / "reference.png", dpi=180)


if __name__ == "__main__":
    main()
