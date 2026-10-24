---
title: Timestral Weather Lab
emoji: 🌦️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Station-level probabilistic forecasting benchmark (Timestral vs baselines vs AROME)
---

# Timestral Weather Lab

This is a small, reproducible benchmark for **short-range probabilistic weather forecasting at French surface stations**. It compares Mistral's [Timestral](https://github.com/mistralai/timestral), a time-series foundation model used zero-shot with **no local fine-tuning**, against persistence, a 24 h seasonal-naive forecast, empirical-quantile climatology and **issued AROME** forecasts. All systems run on frozen case sets from real Météo-France data, and a Gradio explorer presents the saved results.

**Research question.** Can Timestral improve station-level probabilistic forecasts on its own, and when it is conditioned on issued AROME predictions?

> **Status (2026-10-24): initialization complete, results are integration checks only.** The scores below come from a deliberately small smoke suite and a 12-day AROME sample. They show that the pipeline works end to end on real data. They are not performance claims.

## Quick start

```bash
uv sync
uv run weather-lab data fetch    --config configs/smoke.yaml        # SYNOP 2022-2024 (~20 MB), cached once
uv run weather-lab data prepare  --config configs/smoke.yaml        # clean, regularise, freeze manifest
uv run weather-lab evaluate      --config configs/smoke.yaml        # baselines + Timestral-28M, saves run
uv run weather-lab data fetch    --config configs/arome_smoke.yaml  # + SYNOP 2026 and AROME messages (see note)
uv run weather-lab data prepare  --config configs/arome_smoke.yaml
uv run weather-lab evaluate      --config configs/arome_smoke.yaml
uv run weather-lab score --run-dir runs/<suite>/<run-id>            # recompute scores, no model calls
uv run python app.py                                                # explorer on http://127.0.0.1:7860
uv run pytest                                                       # unit tests (no downloads)
uv run pytest -m real_model                                         # opt-in: real Timestral-5M inference
uv run weather-lab export-example runs/<suite>/<run-id> ...         # refresh the offline UI example
```

**AROME note.** Météo-France's public archive keeps only 14 days. The October 2026 runs used here will be gone after early November 2026. After that, `data fetch` for `arome_smoke.yaml` reports those runs as unavailable, and `data prepare` falls back to the packaged station extract in `data/examples/arome-issued-v1/arome_station_extract.parquet`, with provenance. The frozen suite therefore stays reproducible offline.

The app works without any downloads: it replays the cached runs in `data/examples/`, and labels them as cached. Fresh Timestral inference, used only for the exploratory panel, downloads pinned weights from the Hugging Face Hub on first use. No Mistral API key is needed.

## What is implemented

| Area | Implemented | Not implemented (see [roadmap](docs/roadmap.md)) |
|---|---|---|
| Data | SYNOP annual files resolved from data.gouv.fr metadata. Unit conversion (°C, hPa, m/s), meteorological u/v, deterministic de-duplication, 3-hourly axis with NaN gaps, provenance and checksums. | Quality control beyond what the archive provides (it has no flag columns) |
| AROME | Byte-range download of only the T2m/MSLP GRIB2 messages needed (0.025° SP1). Availability evidence from archive `Last-Modified`. Nearest-grid-point extraction with terrain height and land-sea mask recorded. | Wind, ensembles (AROME-EPS), multi-month archive, land-only point rule |
| Systems | Persistence, seasonal naive, climatology (21 empirical quantiles), Timestral-28M univariate / multivariate / + calendar / + issued AROME covariates, raw AROME | 209M and context comparisons (configured, not run), calibration, fine-tuning |
| Protocol | Two versioned suites with frozen manifests, eligibility fixed before scoring, common-case paired comparison, availability accounting | Full 2018–2024 benchmark run |
| Scoring | MAE/RMSE/bias, mean pinball, 80/90% coverage and width, skill with denominators, block-bootstrap paired differences (only with ≥ 10 blocks), warm-tail subset, CPU latency/memory | CRPS, scenario/dependence scores |
| UI | Scorecard (default view), forecast explorer with station map and bands, case links, CSV/manifest downloads, exploratory run labelled outside the benchmark, methods page | Publishing to a Space (needs a paid HF plan; see below) |

## Evaluated case coverage

| Suite | Cases | Stations | Origins | Leads | Variables |
|---|---|---|---|---|---|
| `synop-historical-v1` (smoke) | 96, all eligible | 07630 Toulouse-Blagnac, 07149 Paris-Orly | 00 UTC on days 1 and 15 of each month: 24 in validation 2023, 24 in test 2024 | 3–48 h every 3 h | t, pmer, u, v |
| `arome-issued-v1` (initialization) | 60, all eligible | the two above plus 07110 Brest, 07190 Strasbourg, 07690 Nice | 06 UTC 2026-10-10 … 10-21, each using the same day's 03 UTC run (complete in archive 2.6–2.8 h after initialization) | 3–48 h (run leads 6–51 h) | t, pmer from AROME; Timestral also forecasts u, v |

Smoke baselines are fitted on 2022 only. AROME-suite baselines are fitted on 2022–2024. Out of 768 targets per split and variable, 761 temperature targets were observed in each smoke split. The AROME suite had no missing targets.

## Integration-check results

Each table shows MAE at selected leads and over all leads, computed on the common valid set of each table. The point forecast is the median for quantile systems. Pinball loss and 80% coverage are averaged over all leads, and only for probabilistic systems. Full tables, per station, season and lead, are in the app and in `data/examples/*/metrics_*.csv`.

**Historical smoke suite, 2 m temperature (°C), test 2024, n = 761 (case, lead) pairs**

| System | 6 h | 24 h | 48 h | all | pinball | cov. 80% |
|---|---|---|---|---|---|---|
| persistence | 1.53 | 2.60 | 3.14 | 3.21 | – | – |
| seasonal naive 24 h | 2.34 | 2.60 | 3.14 | 2.77 | – | – |
| climatology (fit 2022) | 2.65 | 3.08 | 2.87 | 3.17 | 1.09 | 0.69 |
| Timestral-28M, t only | 1.12 | 2.42 | 2.37 | 2.23 | 0.78 | 0.80 |
| Timestral-28M, multivariate | 1.08 | 2.30 | 2.39 | 2.18 | 0.76 | 0.82 |
| Timestral-28M, multivariate + calendar | 1.11 | 2.32 | 2.35 | 2.15 | 0.74 | 0.82 |

The validation split (2023) shows the same ordering: Timestral multivariate has an all-lead MAE of 2.11 against 3.32 for persistence, and 80% coverage of 0.83–0.85. For **MSLP** the gain over persistence is small: an all-lead MAE of 3.63–3.71 hPa against 3.79 in test. At 48 h persistence is slightly better (5.05 vs 5.23 hPa). The differences between Timestral variants are small, and this sample cannot resolve them.

**AROME suite, 12 days, 5 stations, test, n = 960**

| System | t 6 h | t 24 h | t 48 h | t all | t pinball | t cov. 80% | MSLP all (hPa) | MSLP cov. 80% |
|---|---|---|---|---|---|---|---|---|
| persistence | 5.67 | 2.90 | 3.44 | 3.89 | – | – | 4.81 | – |
| climatology (fit 2022–24) | 2.80 | 2.61 | 2.63 | 2.65 | 0.88 | 0.75 | 6.11 | 0.80 |
| Timestral-28M mv + calendar | 1.84 | 2.84 | 3.05 | 2.52 | 0.86 | 0.73 | 3.69 | 0.76 |
| **raw AROME** (deterministic) | **1.02** | **1.27** | **1.44** | **1.09** | n/a | n/a | **0.62** | n/a |
| Timestral-28M + AROME covariates | 1.12 | 1.60 | 1.81 | 1.51 | 0.52 | 0.67 | 0.67 | 0.56 |
| ablation: AROME rows NaN over context | 1.83 | 2.83 | 3.05 | 2.51 | 0.86 | 0.73 | 3.69 | 0.76 |

How to read this, honestly:

1. **Raw AROME is clearly the best point forecast** in this October sample. Standalone Timestral is about 2.3× worse on temperature.
2. Conditioning Timestral on issued AROME improves it a lot (t MAE 2.52 → 1.51 °C, MSLP 3.69 → 0.67 hPa). It still **does not beat raw AROME**, and its intervals are **over-confident** (80% coverage of 0.67 for t and 0.56 for MSLP).
3. When the AROME covariate rows are missing over the context, the forecasts are practically identical to the standalone model. Zero-shot, the model only uses the covariate when it can see that series' history.
4. With 6 two-day blocks, no uncertainty intervals are justified. Nothing here says anything about other seasons or extremes. On the warm-tail subset (28 pairs), raw AROME has MAE 0.83 °C and conditioned Timestral 1.42 °C.

**Reproducibility.** `weather-lab score --run-dir …` regenerated all seven score tables byte-for-byte from the saved predictions, for both suites. A full re-run of `evaluate`, with fresh Timestral calls, also reproduced identical score files.

**Runtime** (Apple M2, 4 torch threads, batch 32, context 240 steps):
- Timestral-28M (27.8 M parameters) loads in 0.3 s from cache, or 3.5 s including the 111 MB download.
- A batch of 96 cases takes 0.7–3.1 s, depending on variant.
- Warm single-case latency is 17–19 ms for t-only, 23–26 ms multivariate, and 52–63 ms with calendar covariates, which doubles the input rows.
- Process peak RSS is about 0.5–0.64 GB. A full smoke evaluation takes about 13 s wall time.

**Footprint.** About 444 MB downloaded in total:
- SYNOP 2022–2024 and 2026: 47.9 MB;
- AROME T2m + MSLP message subsets for 12 runs: 264 MB, plus 1.3 MB of constant fields;
- Timestral-28M: 111 MB;
- Timestral-5M (opt-in test only): 21 MB.

Downloading the full 0.025° SP1 files for the same runs would have taken about 6 GB (9 files of about 55 MB per run). The offline example committed to Git is 8.9 MB.

## Design choices worth knowing

- **AROME package.** The brief pointed at the 0.01° packages, but their surface packages carry no MSLP and are about 6× larger per field. The 0.025° SP1 package (the same AROME model, EURW1S40 grid) has T2m and MSLP. Only those messages are fetched, byte-for-byte, with per-message SHA-256.
- **Availability.** A run is used only if every file it needs appears in the public archive (its `Last-Modified` time) before the origin. That is archive evidence, not an operational dissemination log, so the AROME suite is a retrospective experiment under this assumption.
- **Nice.** The nearest AROME grid point to Nice airport is sea (land-sea mask 0). It is kept, and recorded, under the frozen nearest-point rule.
- **Conditioned Timestral.** The covariate row carries observations over the context and the issued forecast over the horizon. This was declared as the primary variant before scoring; the NaN-context version is a pre-declared ablation.
- `clip_nonnegative=False` is passed to Timestral, because temperatures and wind components can be negative.

## Deployment and licences

- `app.py`, `requirements.txt` (exported from `uv.lock`, without the dev group) and the YAML header above are ready for a Hugging Face Gradio Space. Nothing was published. Per the [Spaces overview](https://huggingface.co/docs/hub/spaces-overview), **creating a Gradio Space now requires a paid plan** (PRO for personal accounts), even though CPU Basic hardware has no hourly cost.
- Code: Apache-2.0. Observations and AROME data: Météo-France, [Licence Ouverte v2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence). See `data/examples/*/ATTRIBUTION.json`.
- **Timestral weights use a Modified MIT licence.** It voids all rights if your company's or employer's global monthly revenue exceeds $20 M. Check it before any employer or commercial use.

## Repository map

```text
app.py                      Gradio explorer (reads saved runs only)
configs/                    smoke.yaml, arome_smoke.yaml, benchmark.yaml (full, not yet run)
src/weather_lab/
  data.py                   SYNOP fetch/clean/regularise
  arome.py                  AROME byte-range fetch, availability, station extraction
  suite.py                  origins, cases, eligibility, frozen manifests
  forecast.py               adapter contract, ForecastBatch, Timestral + raw AROME adapters
  baselines.py              persistence, seasonal naive, climatology
  evaluate.py               run a suite, save predictions, score / rescore
  results.py, plots.py      UI data access and Plotly figures
  cli.py, example.py        command line, offline example packaging
benchmarks/manifests/       frozen suite manifests (committed)
data/examples/              cached runs + observation windows + AROME extract for offline replay
docs/                       sources.md, experiment_protocol.md, roadmap.md
tests/                      unit tests; test_real_timestral.py is opt-in (-m real_model)
```

**Adding a model.** Subclass `Adapter` in `forecast.py`: declare `output` (point / quantiles / ensemble) and `inputs`, return a `ForecastBatch`, and register its `kind` in `evaluate.build_adapters`. Scoring and the UI pick it up unchanged. A regional model joins a station suite only through an explicit extraction adapter matching the target definitions.
