# Raw data (not tracked in git)

**You do not need these files to run the project.** Compact processed copies
of everything the analysis uses are committed in `data/processed/` (see the
main README, Data). The raw files are only needed to regenerate those copies
with `python tools/build_processed_data.py`.

These files are excluded from version control (see `.gitignore`) because they
are large (up to 1.5 GB) and are course-supplied data (MMA3001, Monash
University) rather than data we own or may redistribute publicly.

A small, representative sample derived from these files **is** committed at
[`data/processed/occupancy_sample_50k.csv`](../processed/occupancy_sample_50k.csv)
for fast development and unit testing.

**Where to get them:** the MMA3001 course Google Drive folder, https://drive.google.com/drive/folders/1sB4UyZc4c6h67iKfmWcCm0Kv3DSjnCb2 (Monash login required; download and unzip `5EnvSensor_MayToDec2024_180kRows.zip` and `5occupancySensor_MayToDec2024_9MRows.zip`).

To reproduce this folder, place the following files (as supplied in the
"MMA3001 Project Datasets" course materials) here:

| File | Approx. size | Description |
|---|---|---|
| `5occupancySensor_MayToDec2024_9MRows.csv` | 1.5 GB | Occupancy event log, 5 rooms, Nov 2023 – Apr 2026 |
| `5EnvSensor_MayToDec2024_180kRows.csv` | 337 MB | Environmental sensor readings (nested JSON payload), 5 sensors |
| `Sensor ID and Locations.xlsx` | 15 KB | Maps device/sensor IDs to physical rooms (partial coverage — see report, Limitations) |
| `sample_data/` | ~340 MB | Supplementary 2025–2026 batch (PMV, Detailed, Env_data_MSI_Lab) — **not used** in this project; columns for PMV/Detailed are undocumented (see report, Limitations) |

`tools/build_processed_data.py` reads `5EnvSensor_MayToDec2024_180kRows.csv` and
`5occupancySensor_MayToDec2024_9MRows.csv` (plus any BoM CSVs in `bom/`) and writes the
committed copies in `data/processed/`. Every experiment script uses these raw files
instead of the copies whenever they are present here.

## `bom/` — external weather data (not course-supplied)

`data/raw/bom/` holds real Bureau of Meteorology Daily Weather Observations
CSVs for Moorabbin Airport (station 086077, the nearest official BoM
station to Monash Clayton), fetched by `tools/fetch_bom_weather.py` for
the sensor-vs-outdoor-weather comparison (see report §7). This is public
BoM data, not course-supplied. It is gitignored here, but
`tools/build_processed_data.py` copies it to `data/processed/bom/`, which
is committed: re-running the fetch script cannot reproduce it exactly. BoM's bulk historical-download service blocks automated
requests as scraping; only their public monthly pages are used, and only
for their rolling ~15-month free window — re-running the fetch script
later will pull a different window than what the committed
`reports/weather_comparison_results.json` was generated from.
