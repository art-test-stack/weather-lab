# Weather-lab

This is a small, reproducible benchmark for **short-range probabilistic weather forecasting at French surface stations**. It compares Mistral's [Timestral](https://github.com/mistralai/timestral), a time-series foundation model used zero-shot with **no local fine-tuning**, against persistence, a 24 h seasonal-naive forecast, empirical-quantile climatology and **issued AROME** forecasts. All systems run on frozen case sets from real Météo-France data, and a Gradio app presents the saved results.

**Research question.** Can Timestral improve station-level probabilistic forecasts on its own, and when it is conditioned on issued AROME predictions?

> **Status (2026-10-25).** The 28M / 209M × AROME-input matrix is complete on the AROME suite ([results](docs/experiments/conditioning_matrix.md)). Initialization is complete. A context-length experiment has been validated on 2023 and its choice frozen; the full 2024 test has not been run. The smoke and AROME results are integration checks under stated assumptions, not performance claims.

## Quick start

The repository contains code, configuration and documentation only. Data, saved runs, manifests and weights are not committed: `weather-lab bootstrap` fetches and verifies them.

```bash
uv sync
uv run weather-lab bootstrap                 # idempotent: fetch + verify + prepare; reuses the cache
uv run weather-lab-app                       # app on http://127.0.0.1:7860 (runs the same bootstrap at startup)
uv run pytest                                # unit tests, no downloads
uv run pytest -m real_model                  # opt-in: real Timestral-5M inference (~21 MB)
```

`bootstrap` has two modes, selected with `--mode` or by `configs/bootstrap.yaml`:

| Mode | Source | Use |
|---|---|---|
| `official` | data.gouv.fr SYNOP files, the Météo-France 14-day AROME archive, and the Hugging Face Hub (pinned revision, SHA-256 checked) | Regenerates each suite's manifest and checks it against its committed spec in `benchmarks/specs/`. Reproduces saved results with real inference and checks the score hashes. AROME runs that have left the 14-day archive make this mode fail fast, with an explanation. |
| `snapshot` | One immutable archive (URL + SHA-256) holding manifests, prepared data, the AROME station extract, saved runs and evidence files | Verifies every member against its index. Needed for the AROME suites once their runs have rolled out of the archive. |

`auto` (the default) uses `snapshot` when a snapshot URL is configured (`bootstrap.snapshot` in `configs/bootstrap.yaml`, or `WEATHER_LAB_SNAPSHOT_URL` / `WEATHER_LAB_SNAPSHOT_SHA256`), and `official` otherwise. Progress is printed step by step. Failures name the step and say whether a retry can help. The app shows a **Retry setup** screen instead of crashing.

### Pipeline commands

```bash
uv run weather-lab data fetch    --config configs/smoke.yaml        # SYNOP 2022-2024, cached once
uv run weather-lab data prepare  --config configs/smoke.yaml        # clean, regularise, freeze manifest (checked against spec)
uv run weather-lab evaluate      --config configs/smoke.yaml        # baselines + Timestral-28M; saves a run
uv run weather-lab data fetch    --config configs/arome_v2.yaml     # + SYNOP 2026 and AROME messages (14-day archive)
uv run weather-lab data prepare  --config configs/arome_v2.yaml
uv run weather-lab evaluate      --config configs/arome_v2_matrix.yaml --reuse runs/arome-issued-v2/<earlier-run>
                                                                    # reuse verified identical systems, infer the rest
uv run weather-lab score --run-dir runs/<suite>/<run-id>            # recompute scores, no model calls
uv run weather-lab compare --freeze                                 # context-length selection (configs/context/)
uv run weather-lab compare --export-combined runs/synop-ctxval-2023-v1/combined   # one run with all candidates, for the app
uv run weather-lab diagnose arome-inputs --config configs/arome_v2.yaml --case <case-id>
uv run weather-lab diagnose covariates   --config configs/arome_v2.yaml
uv run weather-lab diagnose forecast-only                           # does each checkpoint use future-only AROME?
uv run weather-lab models prefetch [28M 209M]                       # pinned inference files, SHA-256 verified
uv run weather-lab models status                                    # files on disk / loaded models / saved + cached predictions
uv run weather-lab --offline <command>                              # no Hub requests (WEATHER_LAB_OFFLINE=1)
uv run weather-lab arome evidence | observe | refresh-suite         # availability evidence; new suite ids for new dates
uv run weather-lab spec freeze --config <cfg> [--run-dir <run>]     # write benchmarks/specs/<manifest>.yaml
```

`arome refresh-suite` writes a config for whatever dates are currently in the 14-day archive. If those dates differ from the base suite, it uses a **new suite id** (`arome-issued-r<first>-<last>`) and never reuses a frozen one.

## The app

`uv run weather-lab-app` (or the Space) opens on three tabs. *Experiment* sits above the tabs.

- **Overview** (landing): one **forecast block** above the chart, then the result summary, the chart with an aligned error panel (default **absolute error** |forecast − observation|; signed optional), the primary cards, captions, this forecast's table, the CSV download and the map.
  - **Forecast block:** *Station* (default **Paris-Orly**, `07149`), *Forecast issued*, *Weather variable*, *Model (checkpoint)* (Timestral-28M / **Timestral-209M**, identity only), *AROME conditioning* (none; issued forecast + archived AROME history; issued forecast + observed history, substituted; issued forecast only, diagnostic), *Additional feature groups* (recent AROME errors, dew point and cloud cover, solar elevation, tendencies; any compatible combination; none selected = **no additional features**, a valid configuration), *Calendar inputs*, *Observed context (days)* (the one editable history unit; hours and steps shown read-only) and *Forecast horizon (hours)*. Inside the same block, **Advanced** holds feature parameters (AROME-error variables, dew point / cloud fields, tendency lags), comparison overlays, the shaded-range source, the error display and superseded experiments. **Run forecast** comes last.
  - **Composable configuration** (`weather_lab.forecast_config.ForecastConfig`): model identity + conditioning + feature groups with parameters + calendar + context + horizon. Combinations are not separate model choices. Saved benchmark systems map into the same representation from their recorded `run.json` parameters (`from_system`); their identifiers and results are unchanged.
  - **Strictly explicit execution.** Editing any setting (forecast or presentation, Advanced included) only updates the draft: a **Changes pending — click Run forecast** line, read-only conversions, horizon limits and capability / availability messages. It never resolves, loads weights, builds inputs, infers, applies a cached prediction or redraws. **Run forecast** captures the whole draft at click time, validates it, then: same forecast → presentation only, no model run; else the matching saved benchmark system (a shorter horizon is a labelled slice); else the exploratory cache; else inference. Chart, overlays, intervals, error panel, cards, captions, tables and the CSV are applied together. A per-session guard rejects duplicate submissions and keeps the button disabled through rendering, including when controls are edited; a result that arrives after the draft changed is not applied (it stays cached); a failure keeps the previous result and says why. The page load shows the default saved result once.
  - **Why a configuration cannot run** is stated per subject and kind: *unsupported* (the model family cannot take the input: `models.FAMILIES`), *unavailable* (the data do not exist for this case: no AROME run, no archived AROME before this time, no auxiliary observations), *invalid* (horizon, context, empty parameter). Nothing is dropped or substituted. Recent AROME errors need qualifying archived forecasts even without AROME conditioning. AROME conditioning stops where the issued run ends (48 h after these 06 UTC forecast times; the horizon control's maximum follows and any reduction is stated); without it, up to 120 h (app limit).
  - **Result summary:** *Saved benchmark forecast* (run id) or *Exploratory · cached / computed* (outside the frozen benchmark), with the applied model, conditioning, features (non-default parameters shown), calendar, context and the horizon actually shown. It changes only when a result is applied.
  - **Cards:** for a saved result, the benchmark aggregate of that view over the displayed horizon (a slice is labelled; the applied system is always among them). For an exploratory result, **Current forecast · Exploratory**: MAE, 80% coverage and width on the observed leads, or **Not yet observable**, plus the saved systems on the same leads. Benchmark aggregates stay in Benchmark.
  - **Chart:** it plots the whole observation context the model read and opens on the last 48 h. Dotted lines join the last real observation to each forecast's first value. They are a drawing aid only: not in predictions, errors, coverage, exports or caches.
  - **CSV:** the plotted systems of the applied result over the applied horizon, each row labelled saved or exploratory, with the applied configuration. Export filenames hash the CSV contents under `.tmp/codex/forecast-exports/`, so applying different overlays cannot overwrite an earlier download.
- **Benchmark**: a finding sentence computed from the saved scores for the selected setup, the **actual input history** of the compared systems (requested and observed steps per case, cadence, AROME-history construction and coverage, from run provenance; *Unknown* if not recorded, and flagged when contexts differ), sorted comparison bars (reference outlined), and the **lead-time chart**. That chart is recomputed from saved predictions on matched cases, either at each lead or cumulative over all leads up to H, with per-lead sample counts and a data-derived note on which lead ranges favour a system. It shows block-bootstrap intervals of the mean only when there are at least 10 two-day blocks. Scores: MAE, RMSE, pinball, 80/90% coverage (nominal line), 80/90% width; *Station* selects one station or all. *Experiment matrix* lists every input setup × checkpoint as a score, *not run*, *unsupported input* or *not integrated yet*. *Detailed tables and downloads* and *Advanced: filters, availability, uncertainty, timings* hold the tables, CSV downloads, station/season filters, availability, paired differences and CPU cost.
- **Methods**: a table mapping each readable system name to its exact identifier, definitions of the scores, *Runs, versions and integrity*, *Caches and model files* (checkpoints on disk, models loaded in memory, saved and cached predictions), and the full protocol.

Experiments offered (`EXPERIMENTS` in `src/weather_lab_app/labels.py`):

| In the app | Suite | Status |
|---|---|---|
| AROME vs Timestral · Oct 2026 (default) | `arome-issued-v2` | Integration study |
| History length · 2023 validation | `synop-ctxval-2023-v1` | Validation. Shown from a combined run of the 7 / 30 / 90-day candidates, created with `uv run weather-lab compare --export-combined runs/synop-ctxval-2023-v1/combined`. |
| Station history · integration example | `synop-historical-v1` | Integration check |
| AROME vs Timestral · v1 (superseded) | `arome-issued-v1` | Hidden unless *Show superseded experiment versions* is ticked in the forecast block's Advanced |

**Names.** The UI uses readable names; exact identifiers appear in Methods and in chart hover. Identifiers ending in `-cal` mean *calendar* (hour-of-day and time-of-year inputs), **not calibration**. Main names:

| In the app | Identifier |
|---|---|
| AROME | `arome-raw` |
| Timestral 28M | `timestral-28m-mv-cal` |
| Timestral 28M · no time inputs | `timestral-28m-mv` |
| Timestral 28M + AROME (archived AROME history) | `timestral-28m-mv-cal-arome-archived` |
| Timestral 28M + AROME (observed history): station observations stand in for past AROME, a different input | `timestral-28m-mv-cal-arome-substobs` (v1: `…-arome`) |
| Timestral 28M + AROME (forecast only): no AROME history, ignored by the pinned model | `timestral-28m-mv-cal-arome-futureonly` (v1: `…-arome-nanpast`) |
| Timestral 209M, Timestral 209M + AROME | `timestral-209m-mv-cal`, `timestral-209m-mv-cal-arome-archived` |
| Timestral 209M + AROME (observed history / forecast only) | `timestral-209m-mv-cal-arome-substobs`, `timestral-209m-mv-cal-arome-futureonly` |

## Deployment to a Hugging Face Space (prepared, not published)

**Publishing new data (one command, run where the data lives):**

```bash
uv run weather-lab snapshot publish --id wl-snapshot-v5-<date>   # build → upload → verify → pin
git commit -am "Pin wl-snapshot-v5-<date>" && git push             # then run the deploy workflow
```

`snapshot publish` builds the byte-reproducible archive and uploads it with its `.sha256` to the bucket. It then downloads and hashes the remote copy, and rewrites `bootstrap.snapshot` in `configs/bootstrap.yaml`. The previous snapshot becomes the first fallback, and fallbacks that are not in the bucket are dropped. An existing id is never overwritten with different content. It needs a Hugging Face login with write access to the bucket (`hf auth login`); `--dry-run` builds without uploading. The deploy workflow runs `weather-lab snapshot check` first and stops if the pinned snapshot is not published. Only the snapshot *build* has to happen locally: its inputs (saved runs, prepared data, the AROME extract) are git-ignored, and the AROME runs are no longer in Météo-France's 14-day archive.


The Space is built from this repository. It is not a plain mirror, because its README needs Spaces front matter and its root needs an `app_file`:

```bash
uv run weather-lab snapshot build --id <snapshot-id>      # dist/snapshot/<id>.tar.gz + .sha256 (byte-reproducible)
uv run weather-lab-app build-space --out dist/space       # code + generated README front matter + root app.py launcher
```

- The front matter comes from `src/weather_lab_app/space.yaml`. The launcher only adds `src/` to the path and calls `weather_lab_app.main:main`.
- No data or weights are uploaded. The generated front matter asks Spaces to **preload** Timestral-28M and -209M at build time (`preload_from_hub`, pinned full revisions, inference files only) into the default Hub cache, where the loader looks. At startup the app runs `bootstrap`, which fetches the snapshot and verifies the 28M weights. Exploratory forecasts are cached on the attached bucket (`/data/cache/predictions`; public bucket, so publicly readable). What survives restarts and rebuilds: [docs/caching.md](docs/caching.md).
- Snapshot fallback: if the pinned snapshot cannot be fetched or fails its checksum, `bootstrap` tries `bootstrap.snapshot.fallbacks` in `configs/bootstrap.yaml` in order. These are earlier immutable snapshots, each pinned by SHA-256. `data/bootstrap_state.json` records the snapshot that was used (`snapshot.id`, `snapshot.fallback`, `snapshot.primary_error`), and the app shows a one-line notice under the question. The "Retry setup" screen appears only when no candidate can be installed and there are no saved runs.
- `.github/workflows/deploy-space.yml` automates the build and upload. It runs on manual dispatch or a `space-v*` tag, uses the GitHub environment `huggingface-space` (secret `HF_TOKEN`, variable `HF_SPACE_ID`), runs the unit tests, builds `dist/space`, and uploads it with `hf upload --repo-type space --delete "*"`. That makes the Space an exact copy of the build; `.gitattributes` is kept.

**Current deployment** ([ai-testack/weather-lab](https://huggingface.co/spaces/ai-testack/weather-lab)):

- **App.** The layout described in [The app](#the-app); saved results are replayed only.
- **Snapshot.** The data snapshot lives in the public Storage Bucket [`ai-testack/weather-lab-storage`](https://huggingface.co/buckets/ai-testack/weather-lab-storage). `configs/bootstrap.yaml` pins it as `hf://buckets/ai-testack/weather-lab-storage/<id>.tar.gz` with its SHA-256.
  - In the Space the bucket is attached at `/data`, so `bootstrap` reads the file from the mount. Elsewhere (locally, CI) it downloads it through the Hub API. Both paths verify the checksum.
  - Buckets are mutable and unversioned: never overwrite a snapshot file. Upload a new id (`weather-lab snapshot build --id …`, then `hf buckets cp dist/snapshot/<id>.tar.gz hf://buckets/ai-testack/weather-lab-storage/`) and update the pinned URL and SHA-256.
  - The app only reads the bucket, so attaching it **read-only** is safer.
  - The installed artifacts go to the Space's ephemeral disk (`/home/user/app/data`). They are re-extracted from the mounted snapshot at each restart, which takes seconds.
- **Hardware: ZeroGPU** (`zero-a10g`).
  - torch is pinned to **2.13.0**, the newest version ZeroGPU supports (2.8.0–2.13.0), with Python 3.12.12.
  - `src/weather_lab_app/zerogpu.py` imports `spaces` before torch/gradio, as ZeroGPU requires. Only the exploratory forecast runs in a `@spaces.GPU(duration=60)` function: it uses CUDA when a GPU is attached and CPU elsewhere, and its label shows the device.
  - Saved benchmark results are only replayed, never recomputed in the app, and visitors' GPU quota is used only by exploratory runs.
- **torch 2.14.1 → 2.13.0.** This changes forecasts by at most 1.9×10⁻⁶ and scores by at most 4×10⁻⁷.
  - The specs now store exact score hashes **per environment** (torch version, OS, CPU). The torch-2.14.1 references are kept, and torch-2.13.0 references were added for the three app suites. An official-mode run in an environment with no reference is reported as unverified rather than failed.
  - The context-length decision was re-run under torch 2.13.0 and is unchanged (identical pinball values to five decimals).
  - Saved runs in the snapshot were produced with torch 2.14.1, as recorded in each `run.json`.
- **GitHub.** The workflow uses the environment `huggingface-space`. GitHub creates it automatically on first run, and repository-level secrets and variables (`HF_TOKEN`, `HF_SPACE_ID`) are available inside it. Moving `HF_TOKEN` into the environment and adding a required reviewer or a tag-only deployment rule restricts who can trigger an upload with that write token.

## Suites and versions

| Suite | Cases | What it is |
|---|---|---|
| `synop-historical-v1` (smoke) | 96 (Toulouse, Paris-Orly; days 1 and 15 of each month; validation 2023 and test 2024) | Integration check. Baselines fitted on 2022 only. |
| `synop-ctxval-2023-v1` | 1,809 eligible (5 stations, daily 00 UTC, 2023 only) | Context-length validation, 7 / 30 / 90 days. Baselines fitted on 2018–2022. |
| `arome-issued-v2` | 60 (5 stations, 06 UTC 2026-10-10…21, same-day 03 UTC run) | Current AROME comparison. The historical AROME input is explicit per system. |
| `arome-issued-v1` | the same 60 cases | **Earlier version, kept unchanged.** Its conditioned variant substituted station observations for historical AROME. |

Each suite has a committed spec (`benchmarks/specs/*.yaml`) that a regenerated manifest must reproduce. Methods are in [docs/experiment_protocol.md](docs/experiment_protocol.md).

## Results

All tables report MAE on each table's common valid set. Pinball loss and coverage are given for probabilistic systems only, averaged over leads 3–48 h. Units are °C for `t` and hPa for MSLP.

### Context length, 2023 validation (Timestral-28M, multivariate + calendar)

Selection rule, declared before running: lowest mean temperature pinball; ties within 0.005 °C go to the shorter context.

| Context | t pinball (primary) | t MAE | t RMSE | t cov. 80 / 90 | t width 80 | MSLP pinball | MSLP MAE | per case (batch) | warm single case |
|---|---|---|---|---|---|---|---|---|---|
| 7 days (56 steps) | 0.736 | 2.097 | 2.859 | 0.83 / 0.93 | 7.74 | 1.364 | 3.944 | 10.9 ms | 28 ms |
| 30 days (240) | 0.660 | 1.908 | 2.596 | 0.81 / 0.91 | 6.37 | 1.251 | 3.643 | 31.3 ms | 52 ms |
| **90 days (720)** | **0.646** | **1.873** | **2.551** | 0.80 / 0.90 | 6.05 | **1.220** | **3.555** | 82.4 ms | 76 ms |
| climatology (2018–22) | 0.875 | 2.583 | 3.289 | 0.81 / 0.91 | 8.52 | 2.236 | 6.552 | | |
| persistence | n/a | 3.300 | 4.489 | n/a | n/a | n/a | 4.120 | | |

- Counts: 28,708 temperature (case, lead) pairs for the probabilistic metrics; 28,499 for MAE and RMSE.
- **90 days is selected**: ctx90 − ctx30 = −0.0138 °C, 95% block-bootstrap CI [−0.021, −0.006]. The choice is frozen in `configs/frozen/context-selection-2023.yaml`.
- The 2024 test (`configs/benchmark_test_frozen.yaml`) is not yet run. The 2024 smoke cases were inspected earlier and count as exploratory evidence only.
- Details, including per-lead results, wind and resources: [docs/experiments/context_length_2023.md](docs/experiments/context_length_2023.md).

### Issued AROME comparison, `arome-issued-v2` (12 days × 5 stations, n = 960 pairs)

In short (temperature MAE): AROME 1.09 °C; Timestral 28M 2.52 °C; **Timestral 28M + AROME** (archived history) 2.19 °C, about 13% lower error than Timestral alone but about 2.0× AROME's error, with 81% of observations inside its 80% range. Timestral 209M + AROME reaches 1.90 °C. "mv + calendar" below is the app's *Timestral 28M*.

**Complete 28M / 209M matrix** (run `smoke-matrix-v1-20261025T000314-43244569`; full tables, station and lead breakdowns, paired differences and costs in [docs/experiments/conditioning_matrix.md](docs/experiments/conditioning_matrix.md)). Temperature MAE (°C) and 80% coverage, n = 960 per cell:

| Input setup | 28M | 209M |
|---|---|---|
| No AROME input | 2.52 · 73% | 2.42 · 75% |
| AROME forecast + archived AROME history | 2.19 · 81% | 1.90 · 84% |
| AROME forecast + observed history (substituted) | 1.51 · 67% | 1.18 · **18%** (overconfident: 0.65 °C wide 80% range) |
| AROME forecast only (diagnostic) | 2.51 · 73% | 2.43 · 76% |

Reordering or locally perturbing the future-only AROME row changes its normalized input by 3.6–4.7 standard units but the temperature medians by about 0.002 °C for both checkpoints (archived-history control: 0.4–0.8 °C), so neither checkpoint uses future-only AROME values. No uncertainty intervals: 6 two-day blocks < 10.

| System | Historical AROME input | t 6 h | t 24 h | t 48 h | t all | t pinball | t cov. 80% | MSLP all | MSLP cov. 80% |
|---|---|---|---|---|---|---|---|---|---|
| persistence | – | 5.67 | 2.90 | 3.44 | 3.89 | – | – | 4.81 | – |
| climatology (fit 2022–24) | – | 2.80 | 2.61 | 2.63 | 2.65 | 0.88 | 0.75 | 6.11 | 0.80 |
| **raw AROME** (deterministic) | – | **1.02** | **1.27** | **1.44** | **1.09** | n/a | n/a | **0.62** | n/a |
| (A) Timestral mv + calendar | no AROME row | 1.84 | 2.84 | 3.05 | 2.52 | 0.86 | 0.73 | 3.69 | 0.76 |
| (B) + future AROME | missing | 1.83 | 2.83 | 3.05 | 2.51 | 0.86 | 0.73 | 3.69 | 0.76 |
| (C) + future AROME | archived AROME forecasts (18% coverage) | 1.52 | 2.48 | 2.69 | 2.19 | 0.75 | 0.81 | 2.73 | 0.88 |
| Timestral-**209M** mv + calendar | no AROME row | 1.80 | 2.71 | 2.88 | 2.42 | 0.83 | 0.75 | 3.76 | 0.78 |
| Timestral-**209M** (C) | archived AROME forecasts | 1.35 | 2.00 | 2.15 | 1.90 | 0.66 | 0.84 | 1.91 | 0.92 |
| + future AROME (v1 behaviour) | **substituted observations** | 1.12 | 1.60 | 1.81 | 1.51 | 0.52 | 0.67 | 0.67 | 0.56 |

How to read this:

1. **Raw AROME is the best point forecast** in this October sample.
2. The conditioned variant from v1 (1.51 °C) uses station observations as the historical AROME segment. That is a different input from AROME, now labelled explicitly. Its intervals are over-confident.
3. **(B) future-only AROME has no effect.** A +5 °C change of the future AROME temperature moves outputs by at most 1.2×10⁻⁴. The values do reach the network, but the pinned model's masking treats a covariate row with no observed history as not yet started, so it cannot influence a single-block forecast. Traced in [docs/diagnostics/arome_covariates.md](docs/diagnostics/arome_covariates.md).
4. **Timestral-209M** improves on 28M most clearly with archived AROME history (t 2.19 → 1.90 °C, MSLP 2.73 → 1.91 hPa), but stays behind raw AROME. It costs about 5–7× more CPU time per case (≈ 215–240 ms vs 34–54 ms) and about 1.8 GB peak memory.
5. **(C) honest archived history** helps (2.52 → 2.19 °C), increasingly so as history coverage grows (0% → 37% across origins). It remains well behind raw AROME. Its sensitivity to the future covariate can exceed the size of the perturbation at extreme quantiles; this is recorded as an observed instability.
6. **Availability caveat.** Run availability, and the vintage of archived history, rest on the archive's `Last-Modified` timestamps. The provider documents no publication-time semantics, so these are **retrospective results under an archive-timestamp assumption** ([evidence](docs/availability_evidence.md)). 12 days and 6 two-day blocks do not justify uncertainty intervals or seasonal conclusions.

### Historical smoke suite (unchanged, integration check)

Test 2024, temperature, n = 761:

| System | 6 h | 24 h | 48 h | all | pinball | cov. 80% |
|---|---|---|---|---|---|---|
| persistence | 1.53 | 2.60 | 3.14 | 3.21 | – | – |
| climatology (fit 2022) | 2.65 | 3.08 | 2.87 | 3.17 | 1.09 | 0.69 |
| Timestral-28M mv + calendar (30 d) | 1.11 | 2.32 | 2.35 | 2.15 | 0.74 | 0.82 |
| Timestral-209M mv + calendar (30 d) | 1.02 | 2.22 | 2.20 | 2.04 | 0.71 | 0.82 |

The 209M rows are integration checks on the same frozen cases. The size comparison that should inform model choice is the 2023 validation run at the frozen 90-day context, which has not been run yet (see the roadmap).

### Reproducibility checks performed (2026-10-24)

- `score --run-dir` regenerates every score table byte-for-byte. Fresh re-inference reproduces identical score files.
- **Clean source tree.** A tree built only from GitHub-intended files (no data, manifests or model cache; empty `HF_HOME`) was tested in three ways:
  - **Snapshot mode:** fetched the snapshot and weights itself, served the app, and reused the cache on restart.
  - **Official mode:** downloaded SYNOP from data.gouv.fr, regenerated the frozen manifest exactly as specified, and reproduced the spec score hashes with real inference.
  - **Failure handling:** a configuration error and an unreachable snapshot server both showed the retry screen. After the server came up, **Retry setup** completed and the app restarted with full results.

## Next steps

1. **AROME probabilistic baseline**: give AROME an uncertainty range estimated from its historical forecast errors, so Timestral variants are compared with a probabilistic AROME rather than a deterministic one.
2. **Timestral-based residual correction of AROME**: use Timestral to forecast AROME's error and correct it, to test whether Timestral adds information beyond AROME.

The 2024 test of the frozen 90-day context also remains to be run; see [docs/roadmap.md](docs/roadmap.md). The next milestone, testing whether Timestral can improve on AROME (success criteria, AROME-anchored residual design, data inventory, collection plan, go/no-go), is planned in [docs/experiments/arome_improvement_plan.md](docs/experiments/arome_improvement_plan.md); nothing of it has been run.

## Design choices worth knowing

- **AROME package.** The 0.025° SP1 package is used; the 0.01° surface packages carry no MSLP. Only the T2m/MSLP messages needed are fetched, byte-for-byte, with per-message SHA-256.
- **Station extraction.** Nearest grid point. Nice's nearest point is sea; this is recorded.
- **Historical AROME.** Archived forecasts must have been in the archive before both the origin and their own valid time, and missing values are never filled with observations. History uses leads 6–27 h, against 6–51 h in the future segment, which is a documented possible distribution mismatch.
- `clip_nonnegative=False` is passed to Timestral, because temperatures and wind components can be negative.

## Licences

- Code: Apache-2.0.
- Observations and AROME data: Météo-France, Licence Ouverte v2.0.
- **Timestral weights: Modified MIT.** No rights apply if your company's or employer's monthly revenue exceeds $20 M. See [docs/sources.md](docs/sources.md).

## Repository map

```text
configs/                    smoke, arome_v2 + arome_v2_matrix (current), arome_smoke (v1), benchmark, context/, frozen/, bootstrap.yaml
benchmarks/specs/           committed YAML specs of frozen suites (manifests are generated locally)
src/weather_lab/            science: data, suites, models, evaluation, setup
  bootstrap.py              official / snapshot setup, snapshot builder
  specs.py                  spec freeze / manifest + score validation
  data.py, arome.py         SYNOP and AROME acquisition, archived AROME history, refresh-suite
  suite.py                  origins, cases, eligibility, manifests
  forecast.py, baselines.py adapter contract, Timestral / AROME / baseline adapters, exploratory forecast
  models.py                 model families + capabilities, pinned checkpoints, prefetch/offline, loaded-model LRU
  predcache.py, status.py   exploratory prediction cache (content-addressed, atomic); cache status
  evaluate.py, compare.py   run (with verified reuse), score, rescore; cross-run selection
  horizon.py                error by lead / cumulative to H from saved predictions (matched cases)
  diagnostics.py            AROME input audit and covariate sensitivity
  evidence.py               availability evidence and manual first-seen observation
src/weather_lab_app/        app and deployment only (no benchmark logic)
  main.py                   Gradio app (Overview / Benchmark / Methods); runs bootstrap at startup, reads saved runs only
  labels.py                 readable names, descriptions and colours; experiments shown in the app
  results.py, plots.py      UI data access and figures
  space.py, space.yaml      Hugging Face Space build (generated front matter + root launcher)
  cli.py                    `weather-lab-app [serve|build-space]`
.github/workflows/          deploy-space.yml (manual / tag-triggered Space deployment)
docs/                       methods, sources, availability evidence, diagnostics, experiments, roadmap
tests/                      unit tests; test_real_timestral.py is opt-in
```

**Adding a model.** See [docs/adding_a_model.md](docs/adding_a_model.md) (contract, capabilities, and the concrete integration points for TFC t0-gamma, the next candidate). In short: subclass `Adapter`, return a `ForecastBatch` on the benchmark quantile grid, register its `kind` in `evaluate.ADAPTERS` and its capabilities and pinned checkpoint in `weather_lab.models`. Scoring, storage, caching and the app pick it up unchanged. For the UI, add a readable name, description and colour for its identifier in `src/weather_lab_app/labels.py` (`MODELS` and `STYLE`); `tests/test_app_packaging.py::test_labels_cover_every_configured_system` enforces this.
