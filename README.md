# MMA3001 Project — Indoor CO2 Prediction

[![CI](https://github.com/tgup0010-bot/MMA_Final/actions/workflows/ci.yml/badge.svg)](https://github.com/tgup0010-bot/MMA_Final/actions/workflows/ci.yml)

Individual project for MMA3001 (Numerical Methods and Machine Learning), Monash University, 2026 S2.

Author: Tanmay Gupta

## Engineering Problem

Indoor CO2 rises when rooms are occupied and ventilation is insufficient. Above 800–1000 ppm, occupants experience reduced alertness. Most building systems react *after* CO2 is already too high.

This project predicts CO2 concentration **15 minutes ahead** from recent sensor history, so a building management system can pre-emptively increase ventilation. It also runs a full zone-matching analysis to test whether occupancy data is useful — and finds that it is not, for this sensor.

**Status: complete.** All four Week 5 regression methods are implemented and compared. A Week 6 numerical integration analysis is included. The full written report is at [`docs/report/report.md`](docs/report/report.md) and [`docs/report/MMA3001_Project_Report.docx`](docs/report/MMA3001_Project_Report.docx).

## Results

**Best model: SVR** — MAE 8.35 ppm, R² 0.342 on the held-out test set (chronological split, cutoff 11 March 2026).

| Model | MAE (ppm) | RMSE (ppm) | R² |
|---|---|---|---|
| Linear Regression | 9.012 | 25.579 | 0.3581 |
| Decision Tree Regression | 9.353 | 28.708 | 0.1914 |
| **SVR** | **8.350** | 25.894 | 0.3422 |
| Neural Network Regression | 9.350 | 26.015 | 0.3360 |

**Occupancy does not help.** Adding building occupancy as a feature makes no meaningful difference for any model (and actively hurts Decision Tree). This is explained by the zone-matching analysis below.

## Zone-Matching Analysis (Keena's Weekend Spike Method)

Before using occupancy as a feature, a dedicated analysis tested whether sensor `6012002000326` is physically located in any of the 5 monitored occupancy zones. Method: on weekends, when only one zone is occupied, CO2 should spike by 50–200 ppm at a co-located sensor.

Run on the full 9-million-row occupancy dataset:

| Zone | CO2 delta (solo-occupied weekends) | Verdict |
|---|---|---|
| Zone A | +6.2 ppm | Physically negligible — not co-located |
| Zone B | −1.4 ppm | Not co-located |
| Zone C | +1.8 ppm | Not co-located |
| Zone D | −5.3 ppm | Not co-located |

Expected if co-located: 50–200 ppm. All zone-CO2 Pearson correlations are also negative (r ≈ −0.14 to −0.17), consistent with HVAC dilution rather than co-location. **Sensor 0326 is not in any monitored room.**

## Using Every Sensor's Usable Intervals

Following teaching-team feedback, each sensor's record was cut into **unbroken 15-minute intervals**, and every interval of at least 75 minutes (1 h history + 15 min target) was used, from any sensor ([`scripts/multi_sensor_experiment.py`](scripts/multi_sensor_experiment.py), report Section 4.2).

| Sensor | Distinct readings | Usable intervals | Usable rows |
|---|---|---|---|
| 0326 (main) | 29,039 | 649 | 17,317 |
| 0869 (room G.38) | 13,800 | 458 | 7,781 |
| 0241, 0227, 0777 | 1–2 each | 0 | 0 |

- **Three sensors contain no real readings.** They re-send one stale packet (same timestamp, same value) about every 11 minutes, so no interval can be formed.
- **Pooling 0326 + 0869 (+56% training data):** SVR MAE 8.30 → 8.14 ppm, but no model's change is statistically significant (block-bootstrap 95% CIs all span zero).
- **Models do not transfer between rooms:** trained on 0326 and tested on 0869, every model is worse than a naive "CO2 stays the same" forecast; trained on 0869's own data, SVR reaches MAE 6.99 ppm.

## Data

Source: MMA3001 Dataset 2 (Monash Smart Infrastructure Occupancy and Environmental Data).

**Main analysis uses sensor `6012002000326`;** sensor `6012002000869` is added in the multi-sensor experiments above:
- 3 sensors (`0227`, `0777`, `0241`) only ever re-send one stale packet — no genuine readings
- Sensor 0869 has ~3.5 months of genuine readings (Feb–Jun 2024)
- Sensor 0326 has genuine readings from Sep 2025 to Apr 2026, validated against Bureau of Meteorology records: r = +0.84 (temperature), r = +0.65 (humidity) over 222 days

**17,010 rows used** from 84,576 candidates — 80% dropped due to a 636-day gap in genuine readings and missing lag/rolling features.

Raw data files are not committed (see [`data/raw/README.md`](data/raw/README.md)).

## Repository Structure

```
├── src/occupancy/       Python package: CO2 regression + integration models
├── scripts/             Experiment scripts (prediction, integration, demo, analysis)
├── tests/               pytest unit tests (38, all passing)
├── docs/report/         Written report (.md source + .docx submission + charts)
├── docs/MMA3001_Presentation.pptx   13-slide presentation
├── docs/api/            HTML API documentation (pdoc, one page per module)
├── reports/             JSON experiment results + pytest test report (HTML, JUnit XML, text)
└── data/                Raw data (gitignored) + processed samples
```

## Setup

```bash
pip install -e ".[dev]"
```

## Live Demo

```bash
python scripts/co2_demo.py
```

Trains SVR fresh, replays 12 real test-set predictions, then lets you type in your own scenario for a live prediction.

## Running Experiments

```bash
python scripts/co2_prediction_experiment.py   # CO2 regression (all 4 models)
python scripts/co2_integration_analysis.py    # Numerical integration analysis
python scripts/multi_sensor_experiment.py     # Interval audit + pooled / cross-sensor experiments
pytest                                         # Run all 38 tests
```

## Tests and Documentation

The latest test run is committed in [`reports/`](reports/): [`test_report.html`](reports/test_report.html) (pytest-html), [`test_report.xml`](reports/test_report.xml) (JUnit XML) and [`test_report.txt`](reports/test_report.txt) — **38 tests, all passing**. CI re-runs the suite on every push and uploads the same reports as a build artifact.

To regenerate the test report and the API docs after changing the code:

```bash
pytest -v --junitxml=reports/test_report.xml --html=reports/test_report.html --self-contained-html | tee reports/test_report.txt
pdoc -o docs/api occupancy occupancy.co2_integration occupancy.co2_models occupancy.co2_regression \
    occupancy.data_loading occupancy.env_sensors occupancy.evaluation occupancy.external_weather \
    occupancy.preprocessing occupancy.sensor_intervals occupancy.sensor_matching occupancy.sensors \
    occupancy.weather_comparison
```

## AI Use

Claude (Claude Code) was used for data exploration, coding, and drafting throughout. Full disclosure and reflection are in the project report Section 8.

## License

MIT — see [`LICENSE`](LICENSE). Course-supplied raw data is excluded from version control.
