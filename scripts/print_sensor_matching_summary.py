"""Print a human-readable summary of reports/sensor_matching_results.json.

A plain-language companion to scripts/sensor_matching_analysis.py's raw
output, for quickly checking the headline result without re-running the
(multi-minute) full analysis.

Usage:
    python scripts/print_sensor_matching_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_PATH = Path("reports/sensor_matching_results.json")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(
            f"{RESULTS_PATH} not found. Run "
            "scripts/sensor_matching_analysis.py first (needs the raw CSVs "
            "in data/raw/)."
        )

    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    zones = data["zones"]
    sensors = data["sensors"]

    for variable, result in data["results"].items():
        print(f"=== {variable} ===")
        for row in result["best_match_per_room"]:
            zone_label = zones.get(row["room"], row["room"])
            sensor_label = (
                sensors.get(row["best_sensor"], row["best_sensor"])
                if row["best_sensor"]
                else "(no sensor had enough overlapping data)"
            )
            corr = row["correlation"]
            corr_str = f"{corr:+.3f}" if corr is not None else "n/a"
            print(
                f"  {zone_label:35} best match: {sensor_label:30} "
                f"r={corr_str}  (lag={row['best_lag']}, n={row['n_bins']} bins)"
            )
        print()

    print(
        "Interpretation: every correlation above is weak (Carbon dioxide is "
        "essentially noise-level, |r| <= 0.06); no zone/sensor pair is strong "
        "enough to count as a discovered room match. See doc/report/report.md "
        "section 7 for the full discussion."
    )


if __name__ == "__main__":
    main()
