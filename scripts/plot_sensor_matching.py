"""Plot the sensor-matching correlation matrices.

Reads ``reports/sensor_matching_results.json`` (written by
``scripts/sensor_matching_analysis.py``) and draws one heatmap per tested
variable: each cell is the best-lag Pearson correlation between a zone's
occupancy and the *change* in a sensor's reading (see
``occupancy.sensor_matching``). Grey cells had too little overlapping data
(including the three sensors that only re-send one stale packet).

Needs no raw data, so it can be rerun from the committed results alone.

Usage:
    python scripts/plot_sensor_matching.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("reports/sensor_matching_results.json")
OUT_PATH = Path("docs/report/chart10_zone_matching.png")
VARIABLES = ("Carbon dioxide", "Temperature")
COLOR_LIMIT = 0.3

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#e2e1da",  # shows through NaN cells
        "axes.edgecolor": "#898781",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)


def main() -> None:
    data = json.loads(RESULTS_PATH.read_text())
    zones = list(data["zones"])
    sensors = list(data["sensors"])
    zone_labels = [data["zones"][z].replace(" (unidentified)", "") for z in zones]
    sensor_labels = [s[-4:] for s in sensors]

    fig, axes = plt.subplots(1, len(VARIABLES), figsize=(12, 4.6), sharey=True)
    for ax, variable in zip(axes, VARIABLES):
        matrix = data["results"][variable]["matrix"]
        grid = np.array([[matrix[s][z] for s in sensors] for z in zones], dtype=float)
        image = ax.imshow(grid, cmap="RdBu_r", vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT, aspect="auto")
        for i in range(len(zones)):
            for j in range(len(sensors)):
                text = "no data" if np.isnan(grid[i, j]) else f"{grid[i, j]:+.2f}"
                dark = not np.isnan(grid[i, j]) and abs(grid[i, j]) > 0.6 * COLOR_LIMIT
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color="white" if dark else "#0b0b0b")
        ax.set_xticks(range(len(sensors)), sensor_labels)
        ax.set_yticks(range(len(zones)), zone_labels)
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        ax.set_xlabel("Environmental sensor (last 4 digits)")
        ax.set_title(variable)
    fig.suptitle("Zone occupancy vs. change in sensor reading per 15 min: best-lag Pearson r (lags up to ±1 h)", fontweight="bold")
    fig.colorbar(image, ax=axes, shrink=0.85, label="Pearson r")
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
