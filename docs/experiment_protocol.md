# Methods and experiment protocol

## What is being tested

**Research question.** Can Timestral improve short-range probabilistic forecasts at individual weather stations? We ask this both for Timestral on its own and for Timestral conditioned on issued AROME predictions.

This project does **probabilistic weather forecasting at stations**. Timestral returns marginal quantiles: 21 levels for each lead time and variable. These are not coherent weather scenarios. Drawing an independent value at each lead would not produce a physically consistent trajectory, and wind-speed quantiles cannot be built from component quantiles.

| Term | Meaning here |
|---|---|
| Inference | Calling the pretrained Timestral checkpoint, with or without covariates. Every Timestral result in this repository is inference with **no local fine-tuning**. |
| Fine-tuning | Updating Timestral parameters. Not done; the public repository has no training code. |
| Calibration | Post-hoc adjustment of output distributions. Not done yet; this is roadmap step 3. |
| Emulation | Fitting a model to reproduce AROME output. Not done, and kept separate from forecasting verified against observations. |

Timestral and WeatherGenerator are separate models. Nothing here runs, implements or fine-tunes WeatherGenerator.

## Data

- **Observations**: Météo-France *Archive Synop OMM*, annual files (Licence Ouverte v2.0). Station ids are kept as 5-character WMO strings.
- **Variables and units**:
  - `t`: 2 m temperature, K → °C.
  - `pmer`: mean-sea-level pressure, Pa → hPa. Station pressure `pres` is not used.
  - `u`, `v`: 10 m wind components in m/s, derived from the 10-minute mean `ff` and direction `dd`. Gusts are not used. The meteorological convention applies: `dd` is the direction the wind blows *from*, so u = −ff·sin(dd) and v = −ff·cos(dd), and a westerly gives u > 0. Calm (ff = 0) gives u = v = 0. A direction outside [0, 360], or dd = 0 with ff > 0, counts as missing.
- **Cleaning**:
  - Duplicate (station, time) rows: keep the latest `insert_time`, then the last row in file order.
  - Rows off the 3-hourly UTC grid are dropped.
  - The archive has no quality-flag columns; empty fields are missing.
  - Data are placed on a common 3-hourly UTC axis, and missing values stay NaN. History is never filled with later observations.
- **AROME**: 0.025° (EURW1S40) **SP1** package from Météo-France's public 14-day archive.
  - Only the 2 m temperature and MSLP GRIB2 messages at the needed leads are downloaded, via byte ranges. Each message is stored byte-for-byte with its URL, offset, ETag, Last-Modified and SHA-256.
  - The 0.01° packages contain no MSLP and are about 6× larger per field.
  - Station values come from the **nearest grid point** (regular lat/lon index rounding), with no height or lapse-rate correction.
  - Model terrain height is within 8 m of station altitude at all five stations. The nearest point is 0.5–1.1 km from each station.
  - **At Nice the nearest grid point is sea** (land-sea mask 0). This is recorded, and the raw AROME temperature there is a marine grid value.

## Suites (scored separately, never pooled)

### `synop-historical-v1`: historical observation benchmark

- Chronological protocol: fitting/baselines 2018–2022, validation 2023, held-out test 2024 (`configs/benchmark.yaml`).
- **Context-length validation** (`synop-ctxval-2023-v1`, three configs in `configs/context/`): 2023 validation only, 1,809 eligible cases. Under the pre-declared rule (temperature pinball, ties within 0.005 °C go to the shorter context), 90 days was selected and frozen in `configs/frozen/context-selection-2023.yaml`. See [the experiment](experiments/context_length_2023.md). The three candidates can be viewed in the app as *History length · 2023 validation*, from a combined run created with `uv run weather-lab compare --export-combined runs/synop-ctxval-2023-v1/combined` (candidates run separately on one frozen manifest, baselines verified identical, re-scored with the standard scorer). The full 2024 test (`configs/benchmark_test_frozen.yaml`) has not been run.
- **Smoke variant** (`configs/smoke.yaml`): baselines fitted on 2022 only; two stations (Toulouse, Paris-Orly); origins at 00 UTC on days 1 and 15 of each month in 2023 (validation) and 2024 (test). That gives 96 cases, all eligible.
- Every target window (origin + 3 h … + 48 h) lies wholly inside its split. History may extend into the preceding period.
- The **smoke scores are integration checks**, not benchmark evidence.

### `arome-issued-v2` (current) and `arome-issued-v1` (earlier version): issued AROME comparison

Both versions use the same cases. v2 makes the historical segment of the AROME covariate explicit per system (see "Historical AROME inputs" below). v1's scores and manifest are kept unchanged as the earlier experiment version.

- Twelve origins at 06 UTC, 2026-10-10 … 2026-10-21, at all five stations: 60 cases, all eligible.
- Each origin uses that day's **03 UTC run**. A run qualifies only if every file it needs was in the archive (its `Last-Modified` time) by the origin. Initialization time alone proves nothing.
- Leads origin + 3 … + 48 h map to run leads +6 … +51 h, which is the end of the run. Nothing is extrapolated.
- The dates were frozen from archive coverage, with the last date limited by verifying observations through 2026-10-23T21Z, before any forecast was scored.
- Availability evidence comes from the public archive, not an operational dissemination log. The provider documents no publication-time semantics, and the object store's `Last-Modified` records write time; see [availability evidence](availability_evidence.md). This is therefore a **retrospective experiment under an archive-timestamp/latency assumption**, not a claim of exact operational availability.
- Baselines here are fitted on 2022–2024.
- This is a 12-day **integration study**. It says nothing about seasonal, multi-year or rare-event performance.

### Common rules

- Observations valid at or before the origin are inputs. No reporting latency is modelled, so this is historical replay, not an operational backtest.
- **Eligibility** is fixed in the manifest before scoring: at least 80% of the `t` context observed, and at least one `t` observation in the last 24 h. The context-validation suite also requires 80% within each of the last 7, 30 and 90 days, giving one common case set for all candidates. Ineligible cases stay in the availability counts.
- Missing targets are excluded from scores. A failed prediction is counted, never replaced by another model.
- Each suite has a manifest in `benchmarks/manifests/`, generated locally and not committed. Its committed counterpart is a compact YAML spec in `benchmarks/specs/` (protocol hash, case-data hash, case counts, fixed-year source checksums, AROME extract checksum, and optionally the score hashes a re-evaluation must reproduce). A regenerated manifest must match its spec. It records the protocol hash, split boundaries, stations, origins, leads, variables, eligibility, source checksums, a checksum of exactly the observations the cases read, case ids and UI examples. `data prepare` refuses to overwrite a manifest whose protocol or case data changed: a change means a new suite version.
- UI examples are chosen by a fixed rule (earliest eligible test origin per station × season), not for appearance.

## Systems and information sets

| System | Inputs | Output |
|---|---|---|
| `persistence` | last observation at or before the origin | point |
| `seasonal-naive-24h` | the last 24 h profile repeated: value at valid − 24 h·⌈lead/24 h⌉ | point |
| `climatology` | fitting-period observations by station × variable × UTC hour × day-of-year window ±15 d (widened to ±30 d if fewer than 20 samples), empirical quantiles at the 21 Timestral levels, no smoothing | quantiles |
| `timestral-28m-t` | 30 days of `t` history | quantiles |
| `timestral-28m-mv` | 30 days of `t, pmer, u, v` history, forecast jointly at each station | quantiles |
| `timestral-28m-mv-cal` | as above, plus known-future sin/cos hour-of-day and day-of-year | quantiles |
| `arome-raw` | issued AROME run available at the origin, nearest grid point | point (deterministic: no bands, probabilistic metrics N/A) |
| `timestral-28m-mv-cal-arome` (v1) = `…-arome-substobs` (v2) | as `mv-cal`, plus AROME `t` and `pmer` as known-future covariate rows. Over the horizon: the issued forecast. Over the history: **station observations substituted for AROME** (not AROME values). | quantiles |
| `timestral-28m-mv-cal-arome-futureonly` (v2, = v1 `…-nanpast`) | AROME rows missing (NaN) over the history | quantiles |
| `timestral-28m-mv-cal-arome-archived` (v2) | AROME rows over the history hold **archived AROME forecasts** from earlier runs available before the origin and before each value's valid time; missing where none qualifies (18% history coverage on average) | quantiles |

### Names used in the app

The app shows readable names; exact identifiers are listed in its Methods tab and in chart hover. **`-cal` means calendar** (sin/cos hour-of-day and day-of-year inputs), **not calibration**; no system here is calibrated. `timestral-28m-mv-cal` is shown as *Timestral 28M*, `timestral-28m-mv` as *Timestral 28M · no time inputs*, `…-arome-archived` as *Timestral 28M + AROME*, `…-arome-substobs` (v1 `…-arome`) as *Timestral 28M + AROME (observed history)*, and `…-arome-futureonly` (v1 `…-arome-nanpast`) as *Timestral 28M + AROME (forecast only)*. The mapping lives in `src/weather_lab_app/labels.py`.

### Historical AROME inputs

- Historical values are **archived forecasts**, **missing values**, or **substituted observations**, depending on the system. They are never analyses, and missing values are never silently filled.
- Archived values obey a vintage rule: the source run was in the archive before both the origin and the value's own valid time. Run time + lead = valid time is re-checked.
- Archived history uses leads 6–27 h, while the future segment uses 6–51 h. This is not leakage, but it is a possible distribution mismatch; it is described and checked in [the covariate audit](diagnostics/arome_covariates.md).
- With the pinned Timestral model, an AROME row whose history is entirely missing has **no effect** on these single-block forecasts. A +5 °C change of its future values moves outputs by at most 1.2×10⁻⁴. The cause is the upstream key-validity masking, traced in the audit.

The AROME ablation holds the checkpoint, targets, context (30 days = 240 steps) and calendar features fixed. **AROME is the only additional input.** Stations are independent batch items; a shared batch axis creates no spatial interaction.

Timestral is called through `TimestralForecaster.from_pretrained(repo, revision=<pinned sha>)` and `forecast(…, known_covariates=…, clip_nonnegative=False)`. `clip_nonnegative` is off because temperatures and wind components can be negative. `symmetric_averaging` keeps the upstream default (on). Inference runs on CPU in `torch.inference_mode`, with batch size 32.

## Scores

Scores are reported per suite, split, variable, station (or ALL), season (or ALL) and lead (3–48 h or all). Leads of 6, 24 and 48 h are highlighted.

- **MAE, RMSE, bias** in physical units. The point forecast is the median for quantile systems. Each is compared with persistence: skill = 1 − MAE/MAE_persistence, reported with its denominator.
- **Mean pinball loss**: mean over the 21 levels of max(τ(y−q), (τ−1)(y−q)), in physical units. Pinball skill is relative to climatology.
- **Coverage and width** of the central 80% ([q0.10, q0.90]) and 90% ([q0.05, q0.95]) intervals.
- **Common valid set**: every table compares systems on the same (case, lead) pairs, where the target was observed and every compared system produced a forecast. Each system's own coverage is reported separately in the availability table, so no system can improve its score by dropping cases.
- **Paired differences** (model − reference) use a block bootstrap over origin days, in blocks of 2 consecutive origin days with all stations of a day together. A 95% interval is shown only with ≥ 10 blocks. The AROME suite has 6 blocks, so it shows **no interval**.
- **Warm tail**: observed `t` above the station × season 90th percentile of the fitting period. This is not a heatwave definition.
- **Runtime**: batch wall time, warm single-case latency (median of 5), first-load time, and process peak RSS on the measured CPU.
- Raw errors are never summed across °C, hPa and m/s, and different case sets are never ranked together. CRPS is not computed; mean pinball loss is the probabilistic score at initialization.

## Contamination and limits

- A chronological local split does not prove these observations were unseen during Timestral pretraining. Results are reported as **"no local fine-tuning"**, not as an uncontaminated zero-shot benchmark.
- Smoke scores are integration checks. Station-level results say nothing about kilometre-scale spatial forecasting.
- Archived replay assumes observations are available at their validity time.
- The AROME suite covers 12 consecutive October days. It has too few independent blocks for uncertainty statements and no extremes.


## Hourly 2026 Benchmark page

The Benchmark page uses `configs/hourly_2026_v2.yaml`, a new immutable suite from the pinned Orly and Blagnac hourly archive. October study results remain reproducible in their original artifacts and are no longer selectable on this page.

Origins advance by one UTC hour from 1 January 2026 00:00 through each station's last genuine measurement. Each window predicts leads 1–48 hours. Genuine late-2025 measurements provide up to 90 days of January context. Inputs end three hours before the decision, reflecting an explicit replay publication-delay assumption; original publication and revision timestamps are absent. The three-hour bridge is never scored or exported. Missing targets and trailing windows remain visible; no future observations enter model inputs.

The initial display contains the persistence reference only, not model results. Run benchmark executes the selected native Timestral or t0 configuration across the requested cases and saves a separate development run. Direct models compare with persistence over observed 2026 dates. AROME-conditioned or residual models also compare with AROME on their common observed support; the archive currently contains daily 03:00 UTC issues from 2 April, not every operational issue. Each hourly origin selects one available issued run covering its complete horizon. Earlier or uncovered windows remain unavailable, with no stitching or substituted forecasts. Point metrics remain separate by physical variable; overlapping hourly windows are dependent and are labelled development replay rather than held-out confirmation.

The v2 RAG configurations build one-hour sliding candidate windows. Weather analogues remain frozen to pre-2026 observations, with same-station/hour/season filters. AROME-error candidates may use earlier 2026 cases only after their entire outcome window plus the observation delay is strictly earlier than the query decision. Index identities separate old daily and new hourly artifacts; Benchmark requires an hourly index of its pinned archive. Index building is CPU work; model inference still requires the explicit Run action.

Prepare or verify the pinned suite with `uv run weather-lab prepare-hourly-benchmark`. A changed archive or protocol requires a new version; the pinned suite is never overwritten. Publish deployments locally with `uv run weather-lab-app build-space` and `uv run hf upload`, without a GitHub workflow.
