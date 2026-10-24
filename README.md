# Weather-lab

This is a small, reproducible benchmark for **short-range probabilistic weather forecasting at French surface stations**. It compares Mistral's [Timestral](https://github.com/mistralai/timestral), a time-series foundation model used zero-shot with **no local fine-tuning**, against persistence, a 24 h seasonal-naive forecast, empirical-quantile climatology and **issued AROME** forecasts. All systems run on frozen case sets from real Météo-France data, and a Gradio explorer presents the saved results.

**Research question.** Can Timestral improve station-level probabilistic forecasts on its own, and when it is conditioned on issued AROME predictions?

> **Status (2026-10-24).** Initialization is complete. A context-length experiment has been validated on 2023 and its choice frozen; the full 2024 test has not been run. The smoke and AROME results are integration checks under stated assumptions, not performance claims.

## Quick start

The repository contains code, configuration and documentation only. Data, saved runs, manifests and weights are not committed: `weather-lab bootstrap` fetches and verifies them.

```bash
uv sync
uv run weather-lab bootstrap                 # idempotent: fetch + verify + prepare; reuses the cache
uv run weather-lab-app                       # explorer on http://127.0.0.1:7860 (runs the same bootstrap at startup)
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
uv run weather-lab evaluate      --config configs/arome_v2.yaml
uv run weather-lab score --run-dir runs/<suite>/<run-id>            # recompute scores, no model calls
uv run weather-lab compare --freeze                                 # context-length selection (configs/context/)
uv run weather-lab diagnose arome-inputs --config configs/arome_v2.yaml --case <case-id>
uv run weather-lab diagnose covariates   --config configs/arome_v2.yaml
uv run weather-lab arome evidence | observe | refresh-suite         # availability evidence; new suite ids for new dates
uv run weather-lab spec freeze --config <cfg> [--run-dir <run>]     # write benchmarks/specs/<manifest>.yaml
```

`arome refresh-suite` writes a config for whatever dates are currently in the 14-day archive. If those dates differ from the base suite, it uses a **new suite id** (`arome-issued-r<first>-<last>`) and never reuses a frozen one.

## Deployment to a Hugging Face Space (prepared, not published)

The Space is built from this repository. It is not a plain mirror, because its README needs Spaces front matter and its root needs an `app_file`:

```bash
uv run weather-lab snapshot build --id <snapshot-id>      # dist/snapshot/<id>.tar.gz + .sha256 (byte-reproducible)
uv run weather-lab-app build-space --out dist/space       # code + generated README front matter + root app.py launcher
```

- The front matter comes from `src/weather_lab_app/space.yaml`. The launcher only adds `src/` to the path and calls `weather_lab_app.main:main`.
- No data or weights are uploaded. At startup the app runs `bootstrap`, which fetches the snapshot and the pinned 28M weights (111 MB) and verifies both.
- `.github/workflows/deploy-space.yml` automates the build and upload. It runs on manual dispatch or a `space-v*` tag, uses the GitHub environment `huggingface-space` (secret `HF_TOKEN`, variable `HF_SPACE_ID`), runs the unit tests, builds `dist/space`, and uploads it with `hf upload --repo-type space --delete "*"`. That makes the Space an exact copy of the build; `.gitattributes` is kept.

**Current deployment** ([ai-testack/weather-lab](https://huggingface.co/spaces/ai-testack/weather-lab)):

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

| System | Historical AROME input | t 6 h | t 24 h | t 48 h | t all | t pinball | t cov. 80% | MSLP all | MSLP cov. 80% |
|---|---|---|---|---|---|---|---|---|---|
| persistence | – | 5.67 | 2.90 | 3.44 | 3.89 | – | – | 4.81 | – |
| climatology (fit 2022–24) | – | 2.80 | 2.61 | 2.63 | 2.65 | 0.88 | 0.75 | 6.11 | 0.80 |
| **raw AROME** (deterministic) | – | **1.02** | **1.27** | **1.44** | **1.09** | n/a | n/a | **0.62** | n/a |
| (A) Timestral mv + calendar | no AROME row | 1.84 | 2.84 | 3.05 | 2.52 | 0.86 | 0.73 | 3.69 | 0.76 |
| (B) + future AROME | missing | 1.83 | 2.83 | 3.05 | 2.51 | 0.86 | 0.73 | 3.69 | 0.76 |
| (C) + future AROME | archived AROME forecasts (18% coverage) | 1.52 | 2.48 | 2.69 | 2.19 | 0.75 | 0.81 | 2.73 | 0.88 |
| + future AROME (v1 behaviour) | **substituted observations** | 1.12 | 1.60 | 1.81 | 1.51 | 0.52 | 0.67 | 0.67 | 0.56 |

How to read this:

1. **Raw AROME is the best point forecast** in this October sample.
2. The conditioned variant from v1 (1.51 °C) uses station observations as the historical AROME segment. That is a different input from AROME, now labelled explicitly. Its intervals are over-confident.
3. **(B) future-only AROME has no effect.** A +5 °C change of the future AROME temperature moves outputs by at most 1.2×10⁻⁴. The values do reach the network, but the pinned model's masking treats a covariate row with no observed history as not yet started, so it cannot influence a single-block forecast. Traced in [docs/diagnostics/arome_covariates.md](docs/diagnostics/arome_covariates.md).
4. **(C) honest archived history** helps (2.52 → 2.19 °C), increasingly so as history coverage grows (0% → 37% across origins). It remains well behind raw AROME. Its sensitivity to the future covariate can exceed the size of the perturbation at extreme quantiles; this is recorded as an observed instability.
5. **Availability caveat.** Run availability, and the vintage of archived history, rest on the archive's `Last-Modified` timestamps. The provider documents no publication-time semantics, so these are **retrospective results under an archive-timestamp assumption** ([evidence](docs/availability_evidence.md)). 12 days and 6 two-day blocks do not justify uncertainty intervals or seasonal conclusions.

### Historical smoke suite (unchanged, integration check)

Test 2024, temperature, n = 761:

| System | 6 h | 24 h | 48 h | all | pinball | cov. 80% |
|---|---|---|---|---|---|---|
| persistence | 1.53 | 2.60 | 3.14 | 3.21 | – | – |
| climatology (fit 2022) | 2.65 | 3.08 | 2.87 | 3.17 | 1.09 | 0.69 |
| Timestral-28M mv + calendar (30 d) | 1.11 | 2.32 | 2.35 | 2.15 | 0.74 | 0.82 |

### Reproducibility checks performed (2026-10-24)

- `score --run-dir` regenerates every score table byte-for-byte. Fresh re-inference reproduces identical score files.
- **Clean source tree.** A tree built only from GitHub-intended files (no data, manifests or model cache; empty `HF_HOME`) was tested in three ways:
  - **Snapshot mode:** fetched the snapshot and weights itself, served the app, and reused the cache on restart.
  - **Official mode:** downloaded SYNOP from data.gouv.fr, regenerated the frozen manifest exactly as specified, and reproduced the spec score hashes with real inference.
  - **Failure handling:** a configuration error and an unreachable snapshot server both showed the retry screen. After the server came up, **Retry setup** completed and the app restarted with full results.

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
configs/                    smoke, arome_v2 (current), arome_smoke (v1), benchmark, context/, frozen/, bootstrap.yaml
benchmarks/specs/           committed YAML specs of frozen suites (manifests are generated locally)
src/weather_lab/            science: data, suites, models, evaluation, setup
  bootstrap.py              official / snapshot setup, snapshot builder
  specs.py                  spec freeze / manifest + score validation
  data.py, arome.py         SYNOP and AROME acquisition, archived AROME history, refresh-suite
  suite.py                  origins, cases, eligibility, manifests
  forecast.py, baselines.py adapter contract, Timestral / AROME / baseline adapters
  evaluate.py, compare.py   run, score, rescore; cross-run selection
  diagnostics.py            AROME input audit and covariate sensitivity
  evidence.py               availability evidence and manual first-seen observation
src/weather_lab_app/        app and deployment only (no benchmark logic)
  main.py                   Gradio explorer; runs bootstrap at startup, reads saved runs only
  results.py, plots.py      UI data access and figures
  space.py, space.yaml      Hugging Face Space build (generated front matter + root launcher)
  cli.py                    `weather-lab-app [serve|build-space]`
.github/workflows/          deploy-space.yml (manual / tag-triggered Space deployment)
docs/                       methods, sources, availability evidence, diagnostics, experiments, roadmap
tests/                      unit tests; test_real_timestral.py is opt-in
```

**Adding a model.** Subclass `Adapter` in `forecast.py`: declare `output` and `inputs`, return a `ForecastBatch`, and register its `kind` in `evaluate.build_adapters`. Scoring and the UI pick it up unchanged.
