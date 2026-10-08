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
    parser = argparse.ArgumentParser(description="Extract Figure 37's diamond force curves from the local paper PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("directory", type=Path)
    options = parser.parse_args()
    options.directory.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temporary:
        svg = Path(temporary) / "page21.svg"
        subprocess.run(["pdftocairo", "-f", "21", "-l", "21", "-svg", str(options.pdf), str(svg)], check=True)
        root = ET.parse(svg).getroot()
    colors = {
        "rgb(100%, 49.804688%, 5.499268%)": "periodic_simulation",
        "rgb(12.205505%, 46.702576%, 70.506287%)": "experiment"
    }
    curves = {}
    for element in root.iter():
        name = {"source-68": "rest", "source-65": "deformed", "source-29": "specimen"}.get(element.get("id"))
        if name:
            data = element.get("{http://www.w3.org/1999/xlink}href", "")
            extension = "png" if data.startswith("data:image/png;base64,") else "jpg"
            if not data.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
                raise ValueError("Expected an embedded PNG or JPEG from pdftocairo.")
            (options.directory / f"paper-{name}.{extension}").write_bytes(base64.b64decode(data.split(",", 1)[1]))
        path = element.get("d", "")
        color = element.get("stroke", "")
        if color not in colors or not path.startswith("M 4666.601562 4683."):
            continue
        if re.search("[ACHQSTVZachqstvz]", path):
            raise ValueError("Expected a polyline, not a curved PDF path.")
        coordinates = np.array(list(map(float, re.findall(r"-?\d+(?:\.\d+)?", path)))).reshape(-1, 2)
        coordinates[:, 0] = (coordinates[:, 0] - 4666.601562) * 20 / (4865.78125 - 4666.601562)
        coordinates[:, 1] = (coordinates[:, 1] - 4683.945312) * 300 / (5063.085938 - 4683.945312)
        curves[colors[color]] = coordinates.tolist()
    if set(curves) != set(colors.values()):
        raise ValueError("Figure 37 paths do not match this paper version.")
    result = {
        "source": options.pdf.name,
        "page": 21,
        "figure": "37, fourth row (diamond lattice)",
        "method": "PDF vector polylines, calibrated using labeled axis ticks",
        "columns": ["compression_percent", "force_N"],
        "axis_calibration_svg": {"x_0": 4666.601562, "x_20": 4865.78125,
                                 "y_0": 4683.945312, "y_300": 5063.085938},
        "curves": curves
    }
    (options.directory / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
    figure, axis = plt.subplots(figsize=(7, 4))
    for name, values in curves.items():
        coordinates = np.array(values)
        axis.plot(*coordinates.T, label=name.replace("_", " "))
    axis.set(xlabel="Compression (%)", ylabel="Force (N)", xlim=(0, 65), ylim=(0, 350),
             title="Paper Figure 37, fourth row — extracted reference")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(options.directory / "reference.png", dpi=180)
    for strain in (5, 10, 15, 20, 25, 30, 40, 50, 55, 60):
        values = {name: float(np.interp(strain, np.array(curve)[:, 0], np.array(curve)[:, 1]))
                  for name, curve in curves.items()}
        print(f"{strain}%: {values}")


if __name__ == "__main__":
    main()
