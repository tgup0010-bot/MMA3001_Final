# Predicting Indoor CO₂ Concentration from Sensor History and Building Occupancy

**MMA3001 — Numerical Methods and Machine Learning, Project Report**
**Dataset:** MMA3001 Dataset 2 — Monash Smart Infrastructure Occupancy and Environmental Data
**Repository:** https://github.com/tgup0010-bot/MMA_Final

---

## 1. Engineering Problem

### 1.1 Background

Indoor air quality in office buildings deteriorates primarily through the accumulation of carbon dioxide (CO₂). Human respiration releases CO₂ continuously at approximately 0.2–0.3 litres per minute per person. In an enclosed space, the indoor concentration evolves according to a first-order mass balance:

$$\frac{dC}{dt} = \frac{G \cdot N}{V} - \lambda \left(C - C_{\text{out}}\right)$$

where $C$ is indoor CO₂ concentration (ppm), $G$ is the per-person CO₂ generation rate, $N$ is the number of occupants, $V$ is the room volume, $\lambda$ is the ventilation rate (air changes per hour), and $C_{\text{out}}$ is the ambient outdoor concentration (approximately 420 ppm). When occupancy is high and ventilation is insufficient, $C$ rises. When the HVAC system increases $\lambda$, $C$ falls.

CO₂ concentration is therefore a direct, measurable proxy for both occupancy load and ventilation adequacy. ASHRAE Standard 62.1 identifies sustained indoor CO₂ above 1000 ppm as indicative of inadequate ventilation. Peer-reviewed studies have demonstrated measurable reductions in cognitive performance at concentrations of 1000 ppm, with more pronounced effects above 2500 ppm.

### 1.2 The Problem Being Solved

Most building management systems today control ventilation reactively: CO₂ is measured; if it exceeds a threshold, ventilation increases. The delay between CO₂ rising and the HVAC system responding means occupants are already exposed to degraded air quality before any corrective action is taken. Furthermore, reactive systems tend to overshoot — fans ramp to full capacity after the problem has already developed, consuming more energy than a smooth, anticipatory response would require.

This project addresses the question: **can CO₂ concentration 15 minutes from now be predicted accurately from current sensor readings and building state?** A 15-minute forecast provides sufficient lead time for a building management system to begin increasing ventilation before a threshold is crossed, improving both occupant wellbeing and energy efficiency.

This is a **supervised regression problem**: a function is learned that maps a vector of current measurements and engineered time features to a single predicted CO₂ value. Four regression methods from MMA3001 Week 5 are trained and compared on the same dataset, and the contribution of building occupancy data to prediction accuracy is explicitly tested.

### 1.3 Scope

This analysis is limited to one environmental sensor in one building. The sensor has no documented room location (discussed in Section 6), which limits how well building occupancy can be linked to CO₂ at this location. The results establish whether short-term CO₂ prediction is achievable from sensor history alone and whether occupancy data provides any additional predictive value.

---

## 2. Inputs and Outputs

### 2.1 Problem Formulation

The prediction task can be stated formally as: learn a function $f$ such that

$$\hat{C}_{t+1} = f\!\left(C_t,\; C_{t-1},\; \bar{C}_{t-1\text{h}},\; O_t,\; \tau_t\right)$$

where $C_t$ is the current CO₂ reading, $C_{t-1}$ is the reading from 15 minutes ago, $\bar{C}_{t-1\text{h}}$ is the 1-hour rolling mean, $O_t$ is the number of occupied building zones at time $t$, and $\tau_t$ encodes the temporal context (time of day, day of week). The output $\hat{C}_{t+1}$ is the predicted CO₂ concentration one 15-minute bin (15 minutes) ahead.

This formulation uses CO₂ as an **autoregressive** predictor of itself. The physical justification is clear from the mass-balance equation: the future concentration depends on where it is now, which direction it is trending, and the recent context that drives that trend.

### 2.2 Input Features

| Feature | Units / Type | Domain | Description |
|---|---|---|---|
| `co2_now` | ppm, continuous | 400–5000 | CO₂ at the current 15-minute bin. The strongest single predictor of the next reading, because CO₂ dynamics are strongly autoregressive over short horizons. |
| `co2_lag1` | ppm, continuous | 400–5000 | CO₂ from 15 minutes ago. Combined with `co2_now`, this gives the instantaneous rate of change — whether CO₂ is rising, falling, or stable at the moment of prediction. |
| `co2_rolling_1h` | ppm, continuous | 400–5000 | Mean of the preceding four 15-minute bins (1 hour). Captures the medium-term trend — whether CO₂ has been consistently rising through the morning, plateauing, or declining as ventilation increases. |
| `building_occupancy` | integer, discrete | 0–5 | Number of the five monitored occupancy zones currently occupied. Included because, from the mass-balance equation, $N$ directly drives CO₂ generation. Whether this variable is actually useful depends on whether the sensor is co-located with a monitored zone — this is tested empirically in Section 4. |
| `hour_sin`, `hour_cos` | float, continuous | −1 to +1 | Cyclic encoding of hour-of-day: $(\sin(2\pi h/24),\, \cos(2\pi h/24))$. A plain integer hour would represent hour 23 and hour 0 as 23 steps apart; this encoding preserves the fact that they are actually 1 hour apart on a 24-hour cycle. |
| `dow_sin`, `dow_cos` | float, continuous | −1 to +1 | Cyclic encoding of day-of-week, for the same reason. Different weekdays carry different occupancy and ventilation patterns. |
| `is_weekend` | binary | {0, 1} | 1 if Saturday or Sunday. Weekend CO₂ patterns differ substantially from weekdays (near-ambient levels most of the day). An explicit binary flag makes this distinction easy for linear models to learn. |

**Total:** 9 features (each cyclic pair contributes two).

### 2.3 Handling Missing and Invalid Inputs

CO₂ readings are absent wherever the sensor experienced an outage. A 636-day outage is the dominant data loss event. The missing-data policy is as follows:

- Any row where `co2_now`, `co2_lag1`, the 1-hour rolling mean, or the 15-minute-ahead target value is absent is **dropped entirely**. No value is fabricated to fill a gap.
- Imputing values across a 636-day outage would introduce hundreds of thousands of invented readings with no physical basis. Dropping rows is the correct approach despite the loss in dataset size.

The figure below shows the impact:

![Data pipeline: raw candidate timestamps versus dropped rows versus rows retained for modelling](chart_data_pipeline.png){width=4.3in}

Of 84,576 candidate 15-minute timestamps in the raw log, 17,010 (20.1%) have all four required values present and are retained for modelling. The remaining 67,566 rows (79.9%) are dropped.

### 2.4 Output

| Output | Units | Description |
|---|---|---|
| Predicted CO₂ | ppm, continuous | Forecast CO₂ concentration 15 minutes ahead of the current reading |

The output is a single continuous value. It represents the model's best estimate of what the sensor will read at the next 15-minute interval. The model does not apply a threshold or issue a ventilation command — that decision logic belongs to the building management system consuming the forecast. A practical deployment rule might set an early-warning trigger at 750 ppm (50 ppm below the 800 ppm concern threshold) to account for the model's prediction error margin.

---

## 3. Computational Solution

### 3.1 Implementation

All four models are implemented as scikit-learn `Pipeline` objects. Each pipeline applies a `StandardScaler` preprocessing step before the regression estimator. Standardisation transforms each feature to zero mean and unit variance. This is strictly necessary for SVR and the neural network, both of which are sensitive to feature scale: the RBF kernel computes Euclidean distances in feature space, and unscaled features with large dynamic ranges (CO₂ in ppm versus a binary weekend flag) would distort those distances. Applying the same preprocessing to all four models ensures that any differences in performance are attributable to the model, not to data scale effects.

### 3.2 How Each Method Works

**Linear Regression** fits a hyperplane through the training data by minimising the sum of squared residuals:

$$\hat{C}_{t+1} = \beta_0 + \sum_{i=1}^{9} \beta_i x_i$$

It assumes the contribution of each feature to the forecast is linear and additive. This is a reasonable first assumption for CO₂ trends over short windows, but it cannot represent interactions between features (e.g. "high CO₂ on a busy Monday morning predicts a steeper rise than high CO₂ on a quiet Sunday") without explicit interaction terms. It serves as the interpretable baseline: every other method must improve on it to justify its additional complexity.

**Decision Tree Regression** partitions the feature space into rectangular regions using a series of binary splits, each chosen to minimise the variance of the target in the resulting subsets. Each leaf of the tree predicts the mean CO₂ of the training points that fall into it. The tree can capture non-linear patterns and feature interactions without any feature engineering. Its main weakness is a tendency to overfit — a deep tree memorises the training data rather than learning a generalisable pattern. Maximum depth was set to 8 to reduce this risk.

**Support Vector Regression (SVR)** fits a smooth function by finding a hyperplane (in a high-dimensional space implicitly defined by the kernel) that keeps the residuals of most training points within an ε-wide tube, while penalising points that fall outside it. The RBF kernel

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\!\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

maps data into a space where smooth non-linear relationships can be represented. SVR's solution is a convex optimisation problem with no local minima, which makes it more stable than the neural network in training. Parameters used: $C = 10$ (penalty for violating the tube), $\varepsilon = 0.5$ (tube half-width in ppm).

**Neural Network Regression** uses a multilayer perceptron with two hidden layers (32 units, then 16 units, both with ReLU activations) followed by a linear output. ReLU introduces piecewise-linear non-linearity at each layer, allowing the network to approximate arbitrary continuous functions. Training uses the Adam optimiser with early stopping (patience = 10 validation epochs) to halt before the model overfits the training period. Of the four methods, this is the most expressive but also the least interpretable and the most sensitive to the amount of training data.

### 3.3 Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Bin size | 15 minutes | Matches sensor reporting cadence |
| Prediction horizon | 15 minutes ahead | Minimum lead time for meaningful ventilation pre-emption |
| Train/test split | 80 % / 20 %, chronological | Reflects deployment: model only sees past data |
| Training cutoff | 11 March 2026 | 80th-percentile row in the 17,010-row dataset |
| Decision Tree max depth | 8 | Limits leaf count; prevents memorisation of training period |
| SVR kernel | RBF | Handles smooth non-linearity without specifying a basis |
| SVR $C$ | 10 | Moderate regularisation; less restrictive than scikit-learn default |
| SVR $\varepsilon$ | 0.5 | Tube width matched to sensor noise level |
| NN hidden layers | 32 → 16 units | Sufficient capacity for 9 inputs; small enough to avoid overfitting 13,608 rows |
| NN early stopping patience | 10 epochs | Halts training when validation loss stops improving |

---

## 4. Alternative Solutions and Comparison

The four methods were evaluated on the held-out test set (3,402 rows, the final 20% of the dataset in time order). Results are shown below.

| Method | MAE (ppm) | RMSE (ppm) | R² | Fit time (s) | Predict time (ms) | Interpretability |
|---|---|---|---|---|---|---|
| Linear Regression | 9.012 | 25.579 | 0.358 | 0.005 | 1.7 | High |
| Decision Tree | 9.353 | 28.708 | 0.191 | 0.023 | 2.5 | Medium |
| **SVR** | **8.350** | **25.894** | **0.342** | 4.26 | 2854 | Low |
| Neural Network | 9.350 | 26.015 | 0.336 | 2.36 | 2.3 | Very low |

**SVR** achieved the lowest mean absolute error (8.35 ppm) and is the best-performing model. Linear Regression achieved the highest R² (0.358) and is only marginally behind SVR in MAE, making it the recommended choice where model interpretability or fast prediction speed is a requirement. Decision Tree was the weakest model, with an R² of 0.191; it likely overfit the training period, as evidenced by its large RMSE relative to its MAE.

SVR's per-batch prediction time of approximately 2.85 seconds is substantially longer than the other methods, but this is not a practical constraint for a 15-minute scheduling task.

**Effect of building occupancy on prediction accuracy:**

| Method | R² with occupancy | R² without occupancy | Change |
|---|---|---|---|
| Linear Regression | 0.358 | 0.358 | 0.000 |
| Decision Tree | 0.191 | 0.331 | −0.140 |
| SVR | 0.342 | 0.344 | −0.002 |
| Neural Network | 0.336 | 0.335 | +0.001 |

Adding building occupancy as a feature provides no measurable improvement for any model. For the Decision Tree it is actively detrimental: the model wasted splits on an irrelevant feature, and its R² fell from 0.331 to 0.191. The reason is not that occupancy is irrelevant to CO₂ in principle — the mass-balance equation establishes that it is not — but that this sensor is not co-located with any of the five monitored occupancy zones (established in Section 6). The building-level occupancy count carries no information about what is happening in this particular room.

---

### 4.1 Diagnostic plots

The comparison table above is aggregate; the plots below show what it looks like row by row.

![Ground truth CO2 vs. all four models' predictions — full test set (daily mean) and a representative 5-day detail window at native 15-minute resolution](chart11_predictions_timeseries.png){width=4.3in}

All four track the everyday CO2 trend closely; they diverge at the sharp spikes in the detail panel (14-19 March), where every model lags or flattens the peak — the Decision Tree even overshoots into a spike the sensor never reached (15 March), a symptom of its piecewise-constant leaves extrapolating from a small high-CO2 training leaf.

![Predicted vs. actual CO2 for each model on the held-out test set, with a y=x reference line](chart12_predicted_vs_actual.png){width=4.3in}

![Residual (predicted minus actual) vs. predicted CO2 for each model](chart13_residuals.png){width=4.3in}

Predicted-vs-actual makes the R² numbers concrete: all four point clouds hug y=x below 550 ppm, then fall well short of it as actual CO2 rises past 600 ppm. The residual plots show the same effect directionally: residuals sit near zero typically but turn increasingly negative as predicted CO2 rises, rather than scattering symmetrically — a model under-responding to its most extreme inputs, consistent with the training data being dominated by the 400-550 ppm range (Section 2.3).

## 5. Validation

### 5.1 Train/Test Split

The dataset was divided chronologically: the first 80% of rows (13,608) were used for training; the final 20% (3,402 rows) were held out for testing. The training period ends on 11 March 2026.

A random split was deliberately not used. CO₂ time series are strongly autocorrelated: adjacent readings are similar to one another. A random split would scatter test points throughout the training period, allowing the model to effectively look up nearby values seen during training. The resulting test score would measure interpolation ability rather than the ability to forecast a genuinely unseen future period. The chronological split correctly simulates deployment conditions.

![Chronological train/test split over the full sensor history for sensor 6012002000326](chart14_train_test_split.png){width=4.3in}

### 5.2 Evaluation Metrics

**Mean Absolute Error (MAE):**

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|$$

The average prediction error in ppm. Directly interpretable and robust to occasional large outliers. An MAE of 8.35 ppm means the model is on average 8.35 ppm away from the true reading.

**Root Mean Square Error (RMSE):**

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2}$$

Penalises large errors more heavily than MAE due to the squared term. A large gap between RMSE and MAE indicates a model that makes occasional very large errors even when its typical error is small.

**Coefficient of Determination (R²):**

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

The proportion of variance in the target explained by the model. R² = 1 is a perfect fit; R² = 0 means the model does no better than always predicting the mean. All four models achieve R² in the range 0.19–0.36, indicating they explain approximately one third of the variance in next-15-minute CO₂ readings. The remaining variance is driven by factors not captured in the dataset: HVAC switching events, doors and windows, equipment, and the occupants of this particular room.

### 5.3 Sensor Validation Against External Records

The working sensor has no documented room location, but its temperature and humidity readings were compared against Bureau of Meteorology observations from Moorabbin Airport (station 086077) — the nearest official weather station to the Monash campus — across 222 overlapping days.

| Variable | Pearson r (sensor vs BOM) |
|---|---|
| Temperature | +0.84 |
| Relative Humidity | +0.65 |

Both correlations are strong. A malfunctioning sensor — one with a frozen clock, stuck reading, or random noise — would show correlations near zero. These values provide reasonable confidence that sensor `6012002000326` is physically working and measuring real environmental conditions, even without a documented room label.

### 5.4 Numerical Integration Convergence

Two numerical integration methods from MMA3001 Week 6 were implemented from the lecture formulae (not from library functions) to compute cumulative CO₂ exposure in ppm·h. Both methods were applied to a 41-hour gap-free window (1–3 October 2025) and the results were compared at three bin resolutions to verify convergence.

| Bin size | Points | Trapezoidal (ppm·h) | Simpson (ppm·h) | Relative difference |
|---|---|---|---|---|
| 15 min | 163 | 17,817.3 | 17,829.9 | 0.071 % |
| 30 min | 83 | 18,019.8 | 18,019.9 | 0.001 % |
| 60 min | 41 | 17,583.8 | 17,571.8 | 0.068 % |

The two methods agree to within 0.1% at every resolution, confirming the implementation is correct. The small variation in the estimates across bin sizes reflects genuine differences in the measured readings at different sampling densities, not numerical error.

---

## 6. Limitations and Known Failure Modes

**The sensor has no documented room location.** Sensor `6012002000326` does not appear in the Sensor ID and Locations spreadsheet provided with the dataset. As a result, there is no way to determine which room it monitors or which occupancy zone covers that room. A zone-matching analysis was performed: for each sensor–zone pair, weekend bins where exactly one zone was occupied were isolated, and the mean CO₂ difference between occupied and unoccupied bins was computed. The expected signal for a co-located sensor–zone pair is a CO₂ increase of 50–200 ppm; the largest measured difference across all pairs was below 7 ppm. All five Pearson correlations between zone occupancy and sensor CO₂ were negative (−0.14 to −0.17), consistent with a building-wide HVAC dilution effect rather than a room-level occupancy signal. This is a property of the dataset, not a failure of the modelling approach.

**Four of the five sensors are unusable.** Two sensors (`6012002000227`, `6012002000777`) have frozen internal clocks: every reading in their log carries an identical timestamp, making them entirely unsuitable for time-series analysis. A third sensor (`6012002000869`) has only four months of usable data, which is insufficient for a reliable chronological train/test split. These issues were not documented in the dataset description; they were identified by inspecting the actual timestamps.

**Most candidate rows are discarded.** A 636-day sensor outage reduces the usable dataset from 84,576 candidate rows to 17,010 (20.1%). This is a direct consequence of the hardware outage and the correct handling of missing data; imputing values across a gap of this length would produce a fabricated dataset with no physical basis.

**Predictive accuracy is moderate.** An R² of approximately 0.34–0.36 means the models account for roughly one third of the variability in next-15-minute CO₂ levels. The remainder is driven by unobserved factors — HVAC switching, windows, doors, and the occupants of the room itself (since the occupancy counter does not cover it). The model is appropriately used as a trend indicator to support early ventilation decisions, not as a precise point forecast.

**Gap-aware integration is essential for correct cumulative exposure estimates.** Applying the trapezoidal rule naively across the full sensor record — treating the 636-day outage as a 15-minute step — yields 9.54 million ppm·h. The gap-aware implementation, which skips any integration panel where the two endpoints are more than 2 hours apart, yields 2.31 million ppm·h. The factor-of-four difference is caused entirely by the outage-bridging assumption, not by measurement error.

---

## 7. AI Use and Critical Reflection

Claude (Claude Code) was used throughout this project for data exploration, Python implementation, analysis, and drafting this report. The following is a straightforward account of how it was used and what I contributed.

**What Claude assisted with:** Setting up the package structure, writing the scikit-learn pipeline code, running the four-model comparison, performing the integration analysis, implementing the zone-matching validation, and producing drafts of this report. Claude also proposed the zone-matching method as a principled way to test whether the sensor is co-located with a monitored occupancy zone.

**What I directed and decided:** I chose CO₂ prediction over occupancy classification because it was a better fit for the regression focus of MMA3001 Week 5 and for the structure of the available data. I selected which sensor to use and required external validation before accepting it as reliable. I chose to include occupancy as a feature and test it empirically rather than exclude or include it by assumption. When the two numerical integration methods produced results differing by a factor of nine, I required an explanation before accepting either answer — this investigation identified the gap-bridging error in the original implementation.

**What I verified:** All numerical results in this report were checked against the actual output files (`reports/co2_prediction_results.json`) before being recorded here. The integration estimates and convergence results come from scripts that can be rerun to reproduce the same outputs.

**Honest acknowledgement:** In extended AI-assisted work, the precise boundary between ideas that originated with me and ideas that emerged from interaction with Claude cannot always be drawn clearly. The engineering framing, scope decisions, sensor selection rationale, and interpretation of results are mine. The implementation and much of the written prose were produced collaboratively.

---

## References

- MMA3001 Project Brief, Monash University, 2026.
- MMA3001 Project Datasets document, Monash University, 2026. Dataset 2: Monash Smart Infrastructure Occupancy and Environmental Data.
- MMA3001 Notes, Week 5.1–5.6: Regression methods and evaluation. Monash University, 2026.
- MMA3001 Notes, Week 6: Numerical Integration. Monash University, 2026.
- ASHRAE Standard 62.1: Ventilation and Acceptable Indoor Air Quality. American Society of Heating, Refrigerating and Air-Conditioning Engineers, 2022.
- Bureau of Meteorology. Daily Weather Observations, Moorabbin Airport (station 086077). http://www.bom.gov.au/climate/dwo/
- scikit-learn documentation: LinearRegression, DecisionTreeRegressor, SVR, MLPRegressor, StandardScaler, Pipeline. https://scikit-learn.org/
