# Conditioning matrix: Timestral-28M and -209M under four AROME input setups (`arome-issued-v2`)

Run 2026-10-25. Integration study: 12 origin days (10–21 Oct 2026, 06 UTC) × 5 stations = 60 cases, leads
3–48 h, 960 (case, lead) pairs per variable. AROME availability rests on archive timestamps (retrospective;
[evidence](../availability_evidence.md)). Timestral runs as released: no local training, no calibration.

## Design

| | |
|---|---|
| Config | `configs/arome_v2_matrix.yaml` (suite `arome-issued-v2`, same manifest as `configs/arome_v2.yaml`) |
| Run | `runs/arome-issued-v2/smoke-matrix-v1-20261025T000314-43244569` (new run; earlier runs untouched) |
| Spec | `benchmarks/specs/arome-issued-v2.yaml`: added score reference `torch-2.13.0/darwin-arm64\|systems-23179b515a96`; nothing else changed |
| Matched inputs | Same cases, verifying observations, AROME station extract (sha256 `ef21d77e…`), 30-day context (240 steps), calendar inputs, preprocessing and scorer. Within a pair only the checkpoint differs. |
| Checkpoints | `mistralai/Timestral-28M@49b26409…`, `mistralai/Timestral-209M@9874d252…` (full revisions in `weather_lab.models.CHECKPOINTS`) |
| New systems (real inference) | `timestral-209m-mv-cal-arome-substobs`, `timestral-209m-mv-cal-arome-futureonly` |
| Reused systems | 9 systems copied from `smoke-20261024T221003-b4845047` by `evaluate --reuse` after checking the manifest, case data, observations and AROME extract hashes, identical targets and identical system descriptions. Copies verified identical to the source rows. Climatology was refitted (its saved revision string records the fit period) and is identical too. |
| Common set | All 12 systems produced finite forecasts for all 60 cases: no missing cases, n = 960 per variable for every system. |

Input setups (readable names used in the app):

| Setup | Historical segment of the AROME covariate | Future segment |
|---|---|---|
| No AROME input | – (no AROME row) | – |
| AROME forecast + archived AROME history | earlier archived AROME forecasts, available before the origin and before their valid time; missing where none qualifies (about 18 % coverage) | issued run available at the origin |
| AROME forecast + observed history (substituted) | the station's own observations stand in for AROME. A different input construction, not real AROME history; it uses only data at or before the origin, so it is not future leakage | same |
| AROME forecast only (diagnostic) | all missing | same |

Reproduce:

```bash
uv run weather-lab evaluate --config configs/arome_v2_matrix.yaml --reuse runs/arome-issued-v2/smoke-20261024T221003-b4845047
uv run weather-lab score --run-dir runs/arome-issued-v2/smoke-matrix-v1-20261025T000314-43244569   # byte-for-byte
uv run weather-lab diagnose forecast-only                                                       # §4
```

## 1. Results, all leads and stations (n = 960 per cell)

Temperature (°C):

| Setup | Checkpoint | MAE | RMSE | pinball | 80 % cov. | 90 % cov. | 80 % width | 90 % width |
|---|---|---|---|---|---|---|---|---|
| raw AROME (deterministic) | – | **1.085** | **1.417** | n/a | n/a | n/a | n/a | n/a |
| No AROME input | 28M | 2.519 | 3.237 | 0.861 | 0.729 | 0.853 | 7.07 | 9.43 |
| | 209M | 2.416 | 3.124 | 0.825 | 0.752 | 0.865 | 7.05 | 9.48 |
| AROME + archived history | 28M | 2.186 | 2.909 | 0.749 | 0.812 | 0.893 | 7.15 | 9.71 |
| | 209M | 1.901 | 2.544 | 0.655 | 0.840 | 0.922 | 6.78 | 9.30 |
| AROME + observed history | 28M | 1.508 | 1.993 | 0.525 | 0.672 | 0.783 | 3.79 | 5.11 |
| | 209M | 1.185 | 1.579 | 0.519 | **0.182** | **0.229** | **0.65** | **0.87** |
| AROME forecast only (diag.) | 28M | 2.513 | 3.232 | 0.860 | 0.730 | 0.854 | 7.08 | 9.45 |
| | 209M | 2.426 | 3.137 | 0.829 | 0.759 | 0.865 | 7.16 | 9.63 |
| climatology (fit 2022–24) | – | 2.648 | 3.354 | 0.884 | 0.753 | 0.860 | 7.42 | 9.50 |
| persistence | – | 3.887 | 4.869 | n/a | n/a | n/a | n/a | n/a |

Mean-sea-level pressure (hPa):

| Setup | Checkpoint | MAE | RMSE | pinball | 80 % cov. | 90 % cov. | 80 % width |
|---|---|---|---|---|---|---|---|
| raw AROME | – | **0.616** | **0.872** | n/a | n/a | n/a | n/a |
| No AROME input | 28M / 209M | 3.688 / 3.759 | 5.229 / 5.327 | 1.253 / 1.267 | 0.760 / 0.784 | 0.855 / 0.875 | 10.0 / 10.5 |
| AROME + archived history | 28M / 209M | 2.734 / 1.905 | 3.930 / 2.693 | 0.938 / 0.659 | 0.879 / 0.919 | 0.952 / 0.974 | 10.4 / 8.6 |
| AROME + observed history | 28M / 209M | 0.670 / 0.689 | 0.927 / 0.931 | 0.247 / 0.282 | 0.562 / 0.264 | 0.667 / 0.368 | 1.2 / 0.6 |
| AROME forecast only (diag.) | 28M / 209M | 3.688 / 3.790 | 5.234 / 5.364 | 1.253 / 1.277 | 0.764 / 0.784 | 0.857 / 0.871 | 10.0 / 10.7 |

Wind components (m/s; AROME supplies no wind here, so the AROME rows only carry t and MSLP): differences
between setups and checkpoints are 0.01–0.12 m/s in MAE (u: 1.77–1.89, v: 1.97–2.10), all within a few
percent. Full tables: `metrics_point.csv`, `metrics_probabilistic.csv` in the run directory.

## 2. Station and lead breakdowns (temperature MAE, °C)

| Setup / checkpoint | Brest (07110) | Paris-Orly (07149) | Strasbourg (07190) | Toulouse (07630) | Nice (07690) |
|---|---|---|---|---|---|
| raw AROME | 0.83 | 0.99 | 1.18 | 1.24 | 1.19 |
| No AROME, 28M / 209M | 2.73 / 2.66 | 3.36 / 3.14 | 2.56 / 2.46 | 2.58 / 2.49 | 1.36 / 1.33 |
| Archived history, 28M / 209M | 2.03 / 1.65 | 2.88 / 2.47 | 2.40 / 2.22 | 2.28 / 2.02 | 1.34 / 1.16 |
| Observed history, 28M / 209M | 1.07 / 0.88 | 1.92 / 1.00 | 1.71 / 1.40 | 1.77 / 1.50 | 1.07 / 1.14 |
| Forecast only, 28M / 209M | 2.73 / 2.66 | 3.33 / 3.17 | 2.55 / 2.49 | 2.58 / 2.46 | 1.36 / 1.34 |

n = 192 pairs per station (12 forecasts × 16 leads).

| Setup / checkpoint | MAE 6 h | 24 h | 48 h | 80 % cov. 6 h | 24 h | 48 h |
|---|---|---|---|---|---|---|
| raw AROME | 1.02 | 1.27 | 1.44 | n/a | n/a | n/a |
| No AROME, 28M / 209M | 1.84 / 1.80 | 2.84 / 2.71 | 3.05 / 2.88 | 0.83 / 0.78 | 0.65 / 0.73 | 0.67 / 0.72 |
| Archived history, 28M / 209M | 1.52 / 1.35 | 2.48 / 2.00 | 2.69 / 2.15 | 0.87 / 0.83 | 0.75 / 0.87 | 0.77 / 0.82 |
| Observed history, 28M / 209M | 1.12 / 1.02 | 1.60 / 1.42 | 1.81 / 1.73 | 0.73 / 0.23 | 0.60 / 0.18 | 0.57 / 0.13 |
| Forecast only, 28M / 209M | 1.83 / 1.82 | 2.83 / 2.72 | 3.05 / 2.89 | 0.83 / 0.78 | 0.65 / 0.73 | 0.67 / 0.72 |

n = 60 pairs per lead. Every lead and the cumulative view are in the app (Benchmark → lead-time chart).

## 3. Paired differences (matched pairs, `paired_matched.csv`)

Mean differences (model − reference) over the 960 pairs, temperature, °C:

| Pair | abs. error | pinball |
|---|---|---|
| 209M − 28M, no AROME | −0.103 | −0.035 |
| 209M − 28M, archived history | −0.285 | −0.094 |
| 209M − 28M, observed history | −0.323 | −0.006 |
| 209M − 28M, forecast only | −0.087 | −0.031 |
| archived − no AROME, 28M / 209M | −0.333 / −0.515 | −0.112 / −0.171 |
| 28M / 209M archived − raw AROME | +1.101 / +0.816 | – |

**No uncertainty intervals.** The 12 origin days form 6 two-day blocks, fewer than the 10 the project
requires for its block bootstrap (`MIN_BLOCKS_FOR_CI`). The 960 pairs share origins and overlapping
horizons and are not independent samples.

## 4. Forecast-only diagnostic: is the future AROME covariate used?

Perturbations of the future segment of the AROME temperature row only, with masks fixed, on all 60 cases
(`experiments/forecast_only_response/result.json`). The archived-history setup is the control (same
perturbation, history present). Normalized input = the covariate row as the pinned code scales it
(non-causal instance mean / std over observed positions).

| Checkpoint | Setup / perturbation | Normalized input max change | t quantiles max change (°C) | t median mean change (°C) |
|---|---|---|---|---|
| 28M | forecast only / reorder (16 values reversed) | 3.59 | 0.030 | 0.0021 |
| 28M | forecast only / +5 °C at 24 h only | 4.68 | 0.017 | 0.0010 |
| 28M | forecast only / +5 °C uniform | 0.000 | 0.00002 | 0.0000 |
| 28M | archived (control) / reorder | 3.82 | 7.60 | 0.377 |
| 28M | archived (control) / +5 °C at 24 h | 4.05 | 8.48 | 0.128 |
| 209M | forecast only / reorder | 3.59 | 0.052 | 0.0021 |
| 209M | forecast only / +5 °C at 24 h only | 4.68 | 0.035 | 0.0013 |
| 209M | forecast only / +5 °C uniform | 0.000 | 0.00002 | 0.0000 |
| 209M | archived (control) / reorder | 3.82 | 15.23 | 0.838 |
| 209M | archived (control) / +5 °C at 24 h | 4.05 | 9.77 | 0.199 |

- A **uniform shift cannot test use**: instance normalization over the 16 observed future values removes
  it (normalized change 0). Reordering and a single-lead change do alter the normalized input (3.6–4.7
  standard units).
- With the history missing, **both checkpoints essentially ignore the values**: medians move by about
  0.001–0.002 °C, two to three orders of magnitude less than the control. 209M responds slightly more
  (max 0.05 vs 0.03 °C), but not meaningfully.
- The forecast-only scores still differ a little from "no AROME input" (mean absolute median difference
  0.024 °C for 28M, 0.055 °C for 209M): the extra row's *presence* changes the computation, its *values* do
  not. Mechanism (key-validity masking of a row without observed history): see
  [../diagnostics/arome_covariates.md](../diagnostics/arome_covariates.md).

## 5. Reading the results

- **Raw AROME has the lowest point error** for temperature and pressure in this sample overall and at the
  highlighted leads (209M with observed history ties it at 6 h, 1.02 °C). Station exceptions for
  temperature: at **Nice**, whose nearest AROME grid point is over sea, AROME (1.19 °C) is behind 209M with
  archived history (1.16) and both observed-history variants (1.07 / 1.14). At Paris-Orly, 209M with
  observed history is level with it (1.00 vs 0.99). With 12 forecasts per station these are descriptive.
- **With archived AROME history**, 209M has lower error than 28M (t 2.19 → 1.90 °C, MSLP 2.73 → 1.91 hPa)
  and interval coverage at or slightly above nominal (84 % / 92 % for 80 % / 90 %). Without AROME the size effect is small
  for t (−0.10 °C) and slightly negative for MSLP (+0.07 hPa).
- **Observed history** gives the lowest Timestral point errors, but the 209M variant is badly overconfident:
  its 80 % range holds 18 % of temperature observations (0.65 °C wide) and 26 % for MSLP. A covariate
  identical to the target's own past is treated as a near-perfect predictor. Its pinball loss is no better
  than 28M's (0.519 vs 0.525 °C). Use it as a diagnostic of input construction, not as an AROME-conditioned
  forecast.
- Nothing here is evidence about other seasons, extremes or a larger sample; no significance is claimed.

## 6. Costs (Apple M2, CPU, 4 torch threads)

| System | Batch inference, 60 cases | Warm single case | Process peak RSS after |
|---|---|---|---|
| 209M observed history (new) | 11.0 s (12.8 s with first use) | 266 ms | 2.08 GB |
| 209M forecast only (new) | 10.5 s | 256 ms | 2.08 GB |

The evaluation process took 30.4 s wall (2.08 GB peak RSS) because 9 systems were reused. One 209M load
(0.63 s, from the local Hub cache) served both new variants, and no weights were downloaded. The
forecast-only diagnostic (both checkpoints, 60 cases, 12 perturbed runs each) took 118 s.
