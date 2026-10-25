# Context enrichment for Timestral, without fine-tuning (features-v1)

Status (2026-10-25): **run.** This is a controlled inference experiment, not a performance claim. No weights
changed. The checkpoint (Timestral-209M @ `9874d252`), context length (30 days = 240 three-hourly steps),
calendar inputs and preprocessing are fixed; only the feature groups change.

## How Timestral takes extra inputs (checked in the pinned source, `timestral` @ `a534bdb`)

- `forecast(past_covariates=(B, C, T))`: known over the context only. Normalised causally per patch, like the
  targets. NaN is masked; a row with no observed patch is never a valid attention key. `models.FAMILIES`
  previously said Timestral had no past covariates; that was wrong and is corrected.
- `forecast(known_covariates=(B, C, T + H))`: known over context and horizon, normalised with whole-row
  instance statistics.
- **Static metadata does not survive as a constant row.** A constant non-binary row normalises to exactly 0. A
  constant 0/1 row is treated as an indicator and left unscaled. No group uses constant rows.
- **An empty covariate row is not exactly neutral.** It is masked, but its presence switches the model to the
  covariate code path. Measured on Timestral-5M: quantiles move by up to 0.008 on a series around 10
  (`tests/test_features.py`). This matters for the AROME-error group on early October cases, where that history
  is empty.

## Feature groups (`src/weather_lab/features.py`, all version 1)

| Group | Role | Rows | Availability rule | Observed share (Oct suite) | Normalised std |
|---|---|---|---|---|---|
| `resid` A: recent AROME errors | past | obs − archived AROME for t and pmer, plus the archived forecast's lead (h) | the forecast for past valid time v comes from the latest run available before v and before the origin; v ≤ origin, so the observation existed at the origin | 18 % (the 12-run archive starts 10 Oct) | 0.78–0.82 |
| `atmos` B: dew point + cloud | past | SYNOP dew point (°C), total cloud cover (%; 101 → missing) | history ≤ origin only; no future values (no forecast of them was collected) | td 100 %, cloud 61 % | 0.84 / 0.75 |
| `solar` C: solar elevation | known | sin of clear-sky solar elevation at the station | astronomical, known in advance (not an observation) | 100 % | 1.00 |
| `tend` C: tendencies | past | 3 h and 24 h differences of observed t and pmer | from the target history | 97–100 % | 0.73–0.85 |

- **Not added:** wind components (u and v are already targets), day length (almost constant over a 30-day
  window, so it would normalise to a trend the time-of-year calendar already spans), radiation (not in the
  SYNOP archive), future humidity/cloud forecasts (no AROME fields beyond t2m/MSLP were collected, and the
  October runs are partly out of the 14-day archive), spatial context (group D: needs a versioned
  neighbour/grid extraction).
- **Provenance and caching:** feature definitions and versions are stored in each system's `params` in
  `run.json`, with observed shares under `info.feature_inputs`. The auxiliary SYNOP file is recorded with its
  SHA-256 (`data.aux_observations`). The exploratory prediction cache key includes the definitions and the
  actual past-covariate arrays (values and masks).
- **Leakage tests:** `tests/test_features.py` checks that AROME-error rows exist only where a qualifying run
  and a past observation exist, and that auxiliary values after the origin never enter.

## Protocol

| Stage | Data | Use |
|---|---|---|
| Development | `arome-issued-v2`, 12 days in Oct 2026, 60 cases (inspected many times) | diagnostics only; the only place AROME-based groups can run |
| Validation | `synop-ctxval-2023-v1`, daily 00 UTC 2023, 1,809 cases, 182 two-day blocks | selection by the pre-declared rule `configs/features/selection_rule.yaml` |
| Final test | `synop-features-test-2025-v1` (new suite, spec frozen), daily 00 UTC 2025, 1,695 eligible cases; 2024 deliberately not loaded | run once with the frozen selection `configs/features/frozen_selection.yaml` |

- The rule keeps a group only if its temperature-MAE paired difference against the base has a 95 % block-bootstrap
  interval entirely below 0, pressure MAE is at most 2 % worse, and temperature pinball is not worse.
- The AROME-error group cannot be selected: no issued AROME history exists for 2023 or 2025.

## Results

### Validation, 2023 (selection data)

All leads and stations, n = 28,708 pairs for t and pmer (28,738 for u and v), same cases for every system:

| Temperature | MAE | RMSE | Pinball | 80 % / 90 % cov. | 80 % width | Δ MAE vs base [95 % CI] |
|---|---|---|---|---|---|---|
| Persistence | 3.298 | 4.485 | – | – | – | |
| 209M base | 1.847 | 2.523 | 0.639 | 81.2 / 90.8 % | 6.18 | |
| + dew point, cloud | 1.846 | 2.521 | 0.638 | 81.3 / 90.9 % | 6.18 | −0.001 [−0.008, +0.006] |
| + solar elevation | 1.848 | 2.524 | 0.639 | 81.1 / 90.8 % | 6.18 | +0.001 [−0.002, +0.004] |
| + tendencies | 1.873 | 2.554 | 0.647 | 80.8 / 90.9 % | 6.25 | **+0.026 [+0.015, +0.037]** |

- **Pressure (MAE, hPa):** base 3.614; dew point/cloud 3.603 (Δ −0.012 [−0.026, +0.001]; pinball Δ −0.0048
  [−0.0094, −0.0007]); solar 3.610; tendencies 3.683 (Δ **+0.069 [+0.007, +0.131]**).
- **Wind:** all groups within ±0.005 m/s of the base.
- **By station** (t MAE, base / dew point+cloud / solar / tendencies): Brest 1.639 / 1.634 / 1.636 / 1.674;
  Paris-Orly 2.022 / 2.026 / 2.027 / 2.045; Strasbourg 2.216 / 2.219 / 2.220 / 2.252; Toulouse
  2.054 / 2.055 / 2.053 / 2.076; Nice 1.303 / 1.294 / 1.302 / 1.318.
- **By lead group** (t MAE): 3–12 h 1.205 / 1.199 / 1.203 / 1.224; 15–24 h 1.813 / 1.815 / 1.815 / 1.836;
  27–48 h 2.186 / 2.186 / 2.187 / 2.216. Seasons show the same ordering.

**Selection (frozen 2026-10-25T12:17:45Z, before the test): no features.** No group met the rule. Dew point +
cloud is the closest (a small, uncertain temperature gain, and a pressure-pinball improvement whose interval
excludes 0). That is a secondary metric among many comparisons, so it is reported, not selected. Tendencies are
clearly harmful.

### Final test, 2025 (run once)

Run `test-features-test-v1-20261025T121801-3035fcb2`, 1,695 eligible cases (January cases without 30 days
of observed context are ineligible because 2024 was not loaded), n = 27,060 pairs for t and pmer.

- **Systems:** the frozen selection is "no features", so the selected system **is** the base. The test
  therefore measures the base out of sample; it is not a feature comparison. The feature systems were not run
  on 2025, because doing so after seeing validation would turn the test into a second selection round.
- **Temperature:** Timestral-209M (30 days, calendar) MAE 1.920 °C, RMSE 2.608, pinball 0.660, coverage 80.2 %
  (80 %) / 90.4 % (90 %), 80 % width 6.28 °C. Persistence MAE 3.407.
- **By station** (t MAE): Brest 1.844, Paris-Orly 2.169, Strasbourg 2.271, Toulouse 2.180, Nice 1.138.
- **By lead group** (t MAE): 3–12 h 1.275, 15–24 h 1.891, 27–48 h 2.258.
- **Pressure:** MAE 3.360 hPa, coverage 78.3 / 89.8 %. **Wind u / v:** MAE 1.790 / 1.754 m/s.
- **Comparison with validation:** temperature MAE 1.847 on 2023 and 1.920 on 2025, both well calibrated (80 %
  coverage 81.2 % and 80.2 %). These are different years and case sets, so this is descriptive, not a ranking.

### Development diagnostics, October 2026 (AROME suite; not used for selection)

60 cases × 16 leads, n = 960 per variable. Only 6 two-day blocks, so no intervals.

| Temperature | MAE | RMSE | Pinball | 80 % / 90 % cov. | 80 % width |
|---|---|---|---|---|---|
| Raw AROME | 1.085 | 1.417 | – | – | – |
| AROME, bias-corrected (same-lead recent errors, k = 5, 7 days) | 1.052 | 1.386 | – | – | – |
| Timestral 209M | 2.416 | 3.124 | 0.825 | 75 / 87 % | 7.05 |
| 209M + AROME (archived) = base | 1.901 | 2.544 | 0.655 | 84 / 92 % | 6.78 |
| base + recent AROME errors | 1.928 | 2.575 | 0.661 | 84 / 92 % | 6.79 |
| base + dew point, cloud | 1.909 | 2.534 | 0.655 | 83 / 92 % | 6.69 |
| base + solar elevation | 1.848 | 2.469 | 0.637 | 85 / 93 % | 6.76 |
| base + tendencies | 1.996 | 2.658 | 0.685 | 83 / 91 % | 6.84 |

- **Pressure MAE (hPa):** AROME 0.616; bias-corrected 0.649 (worse); base 1.905; errors 1.903; dew
  point/cloud 2.019; solar 1.939; tendencies 2.651.
- **Temperature MAE by station** (AROME / bias-corrected / base / errors / dew point+cloud / solar / tendencies):
  - Brest 0.829 / 0.812 / 1.649 / 1.659 / 1.659 / 1.654 / 1.756
  - Paris-Orly 0.989 / 1.031 / 2.467 / 2.556 / 2.529 / 2.389 / 2.747
  - Strasbourg 1.178 / 1.088 / 2.216 / 2.194 / 2.175 / 2.144 / 2.161
  - Toulouse 1.238 / 1.291 / 2.017 / 2.057 / 2.003 / 1.952 / 2.201
  - Nice 1.192 / 1.039 / 1.157 / 1.174 / 1.180 / 1.104 / 1.116
- **Temperature MAE by lead group:** 3–12 h AROME 0.886, bias-corrected 0.840, base 1.424, solar 1.392; 27–48 h
  AROME 1.149, bias-corrected 1.139, base 2.124, solar 2.060.
- **Recent AROME errors:** with only 18 % of the context covered, adding them made temperature slightly worse
  (+0.027) and pressure unchanged. This is not evidence either way about a longer archive (see
  `arome_improvement_plan.md`).
- **The October solar gain (−0.053 °C) did not replicate on 2023 validation (+0.001 °C).** This is exactly why
  October results are diagnostics only.
- **The bias correction** helps temperature at four stations and hurts pressure. It is fitted on ≤ 7 days of
  errors and evaluated on the same fortnight that supplies them.

## Costs (this Mac, CPU)

- **Validation:** 209M over 1,809 cases took 273 s for the base and 375–466 s with features (past covariates
  add rows); wall 25 min, peak RSS 2.17 GB.
- **October pilot:** 12–19 s per system; wall 74 s.
- **2025 test:** base only, 322 s inference over 1,695 cases; wall 5.5 min, peak RSS 2.05 GB.
- **Downloads:** no checkpoint downloads. `synop_2025.csv.gz` (15.9 MB) was fetched once.

## Limitations

- Twelve autumn days of issued AROME are the only data for the AROME-based groups, so AROME-error features
  remain untested beyond a development pilot.
- The validation and test suites have no AROME inputs, so the selection concerns standalone Timestral-209M.
  Whether a group helps the AROME-conditioned model needs a longer AROME archive.
- Cloud cover is 2–52 % missing depending on the station (automatic stations do not report it). Missing values
  are masked, not filled.
