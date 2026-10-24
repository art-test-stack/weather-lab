# Context-length validation, 2023: Timestral-28M with 7 / 30 / 90 days of history

Run 2026-10-24.
- Rule: `configs/context/selection_rule.yaml`, written before any run.
- Configs: `configs/context/ctx07.yaml`, `ctx30.yaml`, `ctx90.yaml`, derived from `configs/benchmark.yaml`.
- Frozen choice: `configs/frozen/context-selection-2023.yaml`.
- Not-yet-run test config: `configs/benchmark_test_frozen.yaml`.

Reproduce:
- `uv run weather-lab data fetch|prepare --config configs/context/ctx07.yaml`
- `uv run weather-lab evaluate --config configs/context/ctxNN.yaml`, for each of the three configs
- `uv run weather-lab compare --freeze`

## Design

| Item | Setting |
|---|---|
| Suite | `synop-ctxval-2023-v1`: one manifest shared by all three configs (protocol `618bfe111373b76a`) |
| Cases | 00 UTC daily origins 2023-01-01 … 2023-12-29 (target windows inside 2023) × 5 stations = 1,815; **1,809 eligible** |
| Eligibility | Fixed from observations before any prediction: `t` at least 80% observed in **each** of the last 7, 30 and 90 days, and an observation in the last 24 h. Six cases fail the 7-day window. |
| History | 90 days loaded for every config (2022 supplies early-2023 history). Each model reads its own last 56, 240 or 720 steps. |
| Model | Timestral-28M @ `49b26409`, multivariate (t, pmer, u, v), calendar covariates, upstream defaults, batch 32, CPU |
| Baselines | Persistence, 24 h seasonal naive, and climatology fitted on 2018–2022 only. Verified identical across the three runs. |
| Test data | 2024 not loaded, run or scored |
| Primary rule | Mean pinball loss for `t` over the common valid set and all leads 3–48 h. Tie if within 0.005 °C, broken by the shorter context. |

## Result: 90 days selected

| Candidate | steps | **t pinball (primary)** | t MAE | t RMSE | t cov. 80 / 90 | t width 80 / 90 | pmer pinball | pmer MAE | per case (batch) | warm single case | peak RSS after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ctx07 | 56 | 0.736 | 2.097 | 2.859 | 0.831 / 0.928 | 7.74 / 10.94 | 1.364 | 3.944 | 10.9 ms | 28 ms | 0.68 GB |
| ctx30 | 240 | 0.660 | 1.908 | 2.596 | 0.812 / 0.911 | 6.37 / 8.55 | 1.251 | 3.643 | 31.3 ms | 52 ms | 0.83 GB |
| **ctx90** | 720 | **0.646** | **1.873** | **2.551** | 0.802 / 0.901 | 6.05 / 8.07 | **1.220** | **3.555** | 82.4 ms | 76 ms | 1.16 GB |
| climatology (fit 2018–22) | – | 0.875 | 2.583 | 3.289 | 0.812 / 0.906 | 8.52 / 10.79 | 2.236 | 6.552 | | | |
| persistence | – | n/a | 3.300 | 4.489 | n/a | n/a | n/a | 4.120 | | | |

Case counts:
- Pinball, coverage and width use the probabilistic common set: 28,708 `t` (case, lead) pairs.
- MAE and RMSE use the point common set, which also needs persistence and seasonal naive to be valid: 28,499 `t` pairs and 28,499 `pmer` pairs; 28,557 for `u` and `v`.
- Units are °C for `t` and hPa for `pmer`.
- Paired differences in `t` pinball (95% block bootstrap, 182 blocks of 2 consecutive origin days):
  - ctx90 − ctx30 = −0.0138 °C, CI [−0.0213, −0.0064];
  - ctx30 − ctx07 = −0.0756 °C, CI [−0.0890, −0.0627].

The difference between 90 and 30 days exceeds the declared tie tolerance, so **90 days is selected and frozen**.

Selected leads, `t` (°C):

| lead | ctx07 MAE / pinball | ctx30 | ctx90 | persistence MAE | climatology pinball |
|---|---|---|---|---|---|
| 6 h | 1.369 / 0.471 | 1.133 / 0.397 | 1.096 / 0.383 | 1.319 | 0.868 |
| 24 h | 2.032 / 0.715 | 1.912 / 0.660 | 1.874 / 0.647 | 2.058 | 0.843 |
| 48 h | 2.568 / 0.905 | 2.368 / 0.814 | 2.318 / 0.795 | 2.665 | 0.840 |

Secondary outcomes:
- `pmer`, `u` and `v` follow the same ordering at every highlighted lead.
- Longer context gives **narrower** intervals.
- For `t`, ctx90's 80% coverage falls to 0.79 at 24–48 h, slightly below nominal. ctx07 over-covers (0.83–0.85).
- Wind intervals over-cover for all contexts (0.86–0.90 at 24–48 h).

Full per-variable, per-lead, per-station and per-season tables are in `experiments/context_selection/*.csv`.

## Costs (Apple M2, 4 torch threads, batch 32)

| Candidate | Timestral batch wall time (1,809 cases) | process wall time | process max RSS (`/usr/bin/time`) |
|---|---|---|---|
| ctx07 | 19.7 s | 26.5 s | 2.06 GB |
| ctx30 | 56.7 s | 63.9 s | 1.81 GB |
| ctx90 | 149.0 s | 157.6 s | 1.95 GB |

The process maximum includes data loading and scoring. The "peak RSS after" column above is the process-wide peak measured right after the Timestral predictions.

Downloads for this experiment: SYNOP 2018–2021, 25.8 MB. 2022–2023 were already cached, and no new weights were needed.

## Caveats

- This is one checkpoint and one variant: 28M, multivariate + calendar. The 209M comparison has not been run.
- One year of validation with daily origins: the block bootstrap accounts for 2-day dependence but not for seasonal regimes.
- The earlier 2024 smoke cases (24 origins × 2 stations) were inspected during initialization. They are exploratory evidence, not a pristine unseen test. The full 2024 test (`configs/benchmark_test_frozen.yaml`) should be run once, with this frozen choice.
