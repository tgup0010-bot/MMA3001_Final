# Indoor CO₂ Forecasting for Demand-Controlled Ventilation

[![CI](https://github.com/tgup0010-bot/MMA3001_Final/actions/workflows/ci.yml/badge.svg)](https://github.com/tgup0010-bot/MMA3001_Final/actions/workflows/ci.yml)

Individual project for MMA3001 Numerical Methods and Machine Learning, Monash University, 2026 S2. Author: Tanmay Gupta.

**Purpose.** Ventilation systems usually raise outdoor airflow only after indoor CO₂ has crossed a setpoint. This project forecasts CO₂ **15 minutes ahead** from the last hour of sensor readings, zone occupancy and time of day, so a building management system could act earlier. It compares the four Week 5 regression methods against a persistence benchmark, tunes them by time-series cross-validation, measures their cost, and uses the Week 6 trapezoidal and Simpson's rules for cumulative exposure. Data: MMA3001 Dataset 2, Monash Smart Infrastructure occupancy and environmental sensors.

The full report is [`doc/report/report.md`](doc/report/report.md) (Word version: [`MMA3001_Project_Report.docx`](doc/report/MMA3001_Project_Report.docx)).

## Quick start

```bash
git clone https://github.com/tgup0010-bot/MMA3001_Final.git
cd MMA3001_Final
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                        # 90 tests, about 3 seconds
python scripts/co2_demo.py    # train the selected model and try your own inputs
```

No data download is needed: run every command from the repository root and the scripts read the committed data in [`data/processed/`](data/processed/).

## Environment and dependencies

- Python 3.10 or later (CI tests 3.10 and 3.12 on every push).
- Runtime: pandas, NumPy, scikit-learn, Matplotlib, openpyxl. Development: pytest, pytest-html, pdoc, Jupyter. All are declared in [`pyproject.toml`](pyproject.toml) and installed by `pip install -e ".[dev]"`.
- Optional: pandoc, only to rebuild the Word report.

## Repository structure

```
MMA3001_Final/
├── .github/workflows/ci.yml   Runs the tests on every push (Python 3.10 and 3.12)
├── README.md                  This file
├── LICENSE                    MIT licence (code)
├── pyproject.toml             Package metadata and dependencies
├── src/occupancy/             Source code: one module per task (17 modules)
├── scripts/                   Experiments: each writes its results to reports/ and its figures to doc/report/
├── tools/                     Utilities: build the processed data, fetch weather data, rebuild docs
├── tests/                     Unit tests (pytest), against known or hand-computed answers
├── data/
│   ├── processed/             Committed compact data used by every script
│   └── raw/                   Course raw files (not committed; see its README)
├── reports/                   Results (JSON, CSV) and the pytest report (HTML, XML, text)
└── doc/
    ├── report/                Report source, Word file and figures
    ├── api/                   HTML API documentation (pdoc)
    ├── walkthrough.ipynb      Notebook demonstrating the workflow using the package
    └── project_brief/         Assessment brief and dataset description
```

Functions live in `src/occupancy/` as importable, tested modules; scripts and the notebook only call them.

| Module | Purpose |
|---|---|
| `data_loading`, `env_sensors`, `external_weather` | Load and validate the raw occupancy log, the nested-JSON sensor log and Bureau of Meteorology CSVs |
| `processed_data` | Save and load the compact committed data; fall back to it when raw files are absent |
| `preprocessing`, `co2_regression` | 15-minute grids, occupancy count, lag/rolling/calendar features, supervised table |
| `co2_models` | The four Week 5 regression pipelines and the selected (tuned) SVR |
| `evaluation` | Chronological split, MAE/RMSE/R², block-bootstrap confidence intervals |
| `co2_integration` | Trapezoidal (irregular and gap-aware) and Simpson's rules, Richardson check |
| `input_validation` | Reject invalid forecast inputs; flag extrapolation beyond the training range |
| `data_quality` | Hampel outlier screening; ventilation rate from the decay of CO₂ releases |
| `performance` | Repeated timing, peak memory, model size, FLOPs per prediction, arithmetic intensity |
| `sensor_intervals`, `sensor_matching`, `weather_comparison`, `rooms`, `sensors` | Sensor audit, zone matching, sensor-vs-weather check, labels |

## Workflow

Run from the repository root. Each step reproduces the committed results (timing measurements vary from run to run).

| Step | Command | Output | Report section |
|---|---|---|---|
| Baseline comparison | `python scripts/co2_prediction_experiment.py` | `reports/co2_prediction_results.json` | 5.1 |
| Diagnostic plots | `python scripts/generate_diagnostic_charts.py` | charts 11–14, `reports/co2_test_predictions.csv` | 5.2, App. A |
| More data, other rooms | `python scripts/multi_sensor_experiment.py` | `reports/multi_sensor_results.json`, charts 15–16 | 5.3, App. C |
| Tuning, performance, sensitivity | `python scripts/optimisation_and_performance.py` (about 8 min) | `reports/optimisation_results.json`, chart 17 | 4.2, 5.4–5.6 |
| Numerical integration | `python scripts/co2_integration_analysis.py` | `reports/co2_integration_results.json` | App. B |
| Sensor vs weather | `python scripts/weather_comparison_analysis.py` | `reports/weather_comparison_results.json` | 4.2 |
| Zone matching | `python scripts/sensor_matching_analysis.py`, then `plot_sensor_matching.py` | `reports/sensor_matching_results.json`, chart 10 | 6, App. C |
| Weekday vs weekend | `python scripts/weekday_weekend_comparison.py` | `reports/weekday_weekend_results.json` | 5.1 |
| Interactive demo | `python scripts/co2_demo.py` | console | 2.4 |
| Rebuild docs, test report, Word report | `bash tools/build_docs.sh` | `doc/api/`, `reports/test_report.*`, `.docx` | |

## Results

Test period: final 20% of rows in time order (3,402 rows from 11 March 2026); settings chosen by 5-fold time-series cross-validation on the training period only.

| Model | MAE (ppm) | RMSE (ppm) | R² |
|---|---|---|---|
| Persistence (next = current) | 9.89 | 29.34 | 0.155 |
| Linear Regression | 9.01 | 25.58 | 0.358 |
| Decision Tree (tuned) | 8.93 | 25.32 | 0.371 |
| Neural Network (tuned) | 8.86 | 25.22 | 0.376 |
| **SVR, tuned: C = 100, ε = 2 ppm, γ = 0.03 (selected)** | **8.11** | 25.30 | 0.372 |

- **The selected SVR beats every alternative**, with 95% block-bootstrap intervals excluding zero (by 1.78 ppm over persistence and 0.91 ppm over Linear Regression), and is best in all five cross-validation folds.
- **No model anticipates sudden rises.** On the 63 test rows where CO₂ rose by more than 30 ppm in 15 minutes, every model predicted none of the rise. The forecast suits gradual, early ventilation changes, not replacing a peak alarm.
- **Occupancy does not help** this sensor (R² changes by at most 0.002): a data-driven search found no link between it and any monitored zone.
- **Models do not transfer between rooms;** each room needs its own training data.
- **Cost:** SVR needs about 500,000 operations per forecast (0.3 ms) against 36 for Linear Regression, which remains the interpretable fallback.
- **Physics check:** the decay of 87 CO₂ releases gives a ventilation rate of 3.9 per hour (time constant about 15 minutes).

## Data

- **`data/processed/`** (committed, about 12 MB): every CO₂, temperature and humidity reading of the five sensors, the occupancy log on a 15-minute grid, each CO₂ reading's server receive time, and the Bureau of Meteorology daily observations (Moorabbin Airport, station 086077). Built by `python tools/build_processed_data.py`; each table reads back identical to the raw-file version, and a test fails if any is missing.
- **`data/raw/`** (not committed): the course files are 1.6 GB and 353 MB, beyond GitHub's 100 MB file limit, and are course-supplied rather than ours to redistribute. [`data/raw/README.md`](data/raw/README.md) lists them. When they are present, every script uses them instead of the processed copies.
- Main analysis: sensor `6012002000326`, genuine readings 7 September 2025 to 22 April 2026. Three of the five sensors only re-send one stale packet and contain no genuine readings.

## Testing and documentation

- **Tests:** `pytest` runs 90 tests on synthetic or hand-checkable inputs: integration of known functions and the order of accuracy (error falls 4× and 16× as the step halves), hand-computed metrics, the decay fit on a manufactured exponential, outlier detection, input validation, data loading edge cases and the committed data. Latest report: [`reports/test_report.html`](reports/test_report.html) (also XML and text).
- **API documentation:** generated by pdoc from the Google-style docstrings, in [`doc/api/`](doc/api/) and online at https://tgup0010-bot.github.io/MMA3001_Final/.
- **Walkthrough:** [`doc/walkthrough.ipynb`](doc/walkthrough.ipynb) runs the workflow end to end by importing the package.
- Run `bash tools/build_docs.sh` after changing code or the report, so documentation and code stay in sync.

## AI use

Claude Code (Anthropic) and the Claude chat interface were used throughout: Claude wrote most of the code, ran the analyses and drafted most of the report. The engineering direction and decisions were mine (problem choice, sensor choice and its external validation, testing occupancy rather than assuming it, acting on teaching-team feedback, reporting negative results as they are), and every reported number was checked against the output files and the data. AI errors found during checking are listed in Section 7 of the report, which is the full reflection.

## Licence

Code: MIT, see [`LICENSE`](LICENSE). The licence does not cover the course-supplied raw data (MMA3001, Monash University), which is not in this repository. The processed data in `data/processed/` is derived from it for reproducibility of this coursework.
