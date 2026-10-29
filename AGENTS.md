# Agent instructions: Weather-lab

## Ground rules

- **Frozen suites.** Manifests (`benchmarks/manifests/*.json`) are generated locally and not committed; their committed counterpart is `benchmarks/specs/<manifest>.yaml`, which a regenerated manifest must reproduce. Never rewrite a spec or a manifest once scores exist. A change to dates, stations, eligibility, preprocessing or leads means a new suite version, with a new `suite`/`manifest` value in a new config. `data prepare` refuses silent changes. Keep it that way.
- **Code-only repository.** `*.json`, `*.csv`, `*.parquet`, data, runs and weights are git-ignored on purpose. Artifacts come from `weather-lab bootstrap` (official sources or the immutable snapshot). Spaces metadata lives in `src/weather_lab_app/space.yaml` and is written only into the Space build (`weather-lab-app build-space`).
- **Historical AROME.** Choose and label the historical segment explicitly: archived forecasts (vintage-checked), missing, or substituted observations. Never fill it silently.
- **No pooling.** Never rank systems across suites, splits, variables or case sets. Never sum errors across °C, hPa and m/s.
- **Labels.** Smoke and AROME-initialization scores are integration checks. Timestral results are inference with no local fine-tuning. Calibration, fine-tuning and emulation are different things and must be labelled as such. Timestral is not WeatherGenerator, and it is a model used here, not the project name.
- **No leakage.** Adapters receive a `CaseBatch` holding history at or before the origin, known-future covariates (calendar, AROME runs available at the origin) and nothing else. Targets stay in `evaluate.make_batch`'s second return value and must never reach an adapter.
- **No fabricated results.** Missing inputs produce unavailable cases or failed predictions in the availability table, never a fallback to another model. Deterministic systems get no bands and no probabilistic metrics. Synthetic fixtures are for tests only.
- **Model selection.** Choose models and contexts on validation only, then score test once.

## Commands

```bash
uv sync
uv run pytest                      # fast, no network
uv run pytest -m real_model        # downloads Timestral-5M (~21 MB)
uv run weather-lab data fetch|prepare --config configs/<cfg>.yaml
uv run weather-lab evaluate --config configs/<cfg>.yaml
uv run weather-lab score --run-dir runs/<suite>/<run-id>   # must report "byte-for-byte"
uv run weather-lab evaluate --config configs/<cfg>.yaml --reuse runs/<suite>/<run>   # copy verified identical systems
uv run weather-lab bootstrap       # idempotent setup (official or snapshot mode)
uv run weather-lab models prefetch|status          # pinned checkpoint files; cache status (add --offline: no Hub)
uv run weather-lab features list|select            # context-enrichment groups; apply the pre-declared selection rule
uv run weather-lab-app            # app: Overview / Benchmark / Methods (weather-lab-app build-space: Space content)
```

## Conventions

- Python 3.12 with `uv`. Dependencies are pinned in `uv.lock`. Regenerate `requirements.txt` with `uv export --no-dev --no-hashes --no-emit-project -o requirements.txt` whenever `uv.lock` changes.
- torch is pinned to 2.13.0 for ZeroGPU (supported 2.8.0–2.13.0); after changing torch, add score references with `weather-lab spec add-reference` (exact hashes are per torch/OS/CPU).
- Timestral is pinned to commit `a534bdb` and the checkpoint revisions in `configs/*.yaml` / `models.CHECKPOINTS` (full 40-character revisions + SHA-256; never `main`). Bump them deliberately and record the change in `docs/sources.md`.
- Caching (`docs/caching.md`): load weights only through `weather_lab.models` (shared files per revision, one loaded object per revision/device/dtype, request data never stored on it). Exploratory forecasts go through `weather_lab.predcache` and are never scored. On ZeroGPU never keep GPU-resident models between calls.
- New model families: follow `docs/adding_a_model.md` (capabilities in `models.FAMILIES`, adapter in `evaluate.ADAPTERS`, benchmark quantile grid or an explicit recorded mapping). Do not add a checkpoint above the download budget to bootstrap `weights` or `space_preload` without saying so.
- Bulk data (`data/raw`, `data/processed`), `runs/`, `data/examples/*/`, `data/evidence/`, `dist/` and weights are git-ignored. Refresh the offline replay with `weather-lab export-example`, then publish it with `weather-lab snapshot build` (record the URL + SHA-256 in `configs/bootstrap.yaml`).
- Keep scientific logic (data, suites, models, evaluation, setup) in `src/weather_lab/`. The app and deployment live in `src/weather_lab_app/`, which may import `weather_lab`, never the reverse. UI callbacks only read saved results. On ZeroGPU only the exploratory forecast may hold a GPU (`zerogpu.gpu`); import `weather_lab_app.zerogpu` before torch/gradio. The exploratory panel never writes to `runs/`.
- **Explorer state.** Two states: the draft (every forecast-block control, presentation included) and `applied` (a gr.State). Control edits reach only `forecast_state.draft_feedback` (pending line, conversions, limits, capability/availability issues): never resolution, model loading, input building, inference, cache lookup or redraw. Only the Run button's chain (`begin_forecast` → `run_captured` / `run_forecast` → `apply_if_current` → `render`) and the page load write `applied` or the Overview display; `tests/test_explicit_run.py` checks the event graph. `run_forecast` validates (`forecast_config.check`), reuses the applied predictions for presentation-only changes, then saved match (`saved_match` on mapped configurations), exploratory cache, inference. `begin_forecast` snapshots the click's draft and claims a per-session busy guard until rendering finishes; edits cannot re-enable Run and duplicate submissions are rejected. `apply_if_current` drops results whose draft is no longer current. Labels come from the applied result, never from the controls. The observation-to-forecast bridge is drawing only (never in predictions, errors, exports or caches). Exploratory cards never reuse benchmark aggregates. Contexts shown in Benchmark come from run provenance (`results.input_history`).
- **Forecast configuration** (`weather_lab.forecast_config`): model identity, AROME conditioning, feature groups + parameters, calendar, context, horizon; never encode combinations as model ids in the app. Saved systems map in through `from_system` (recorded params only; another revision or feature version → no match). Issues are `unsupported` (family capability), `unavailable` (data) or `invalid`; never drop a selected feature, substitute a model or replace archived AROME with observations.
- **Feature groups** (`weather_lab.features`): a model-independent pipeline (`build_inputs` → `FeatureInputs`); adapters translate it (Timestral: past rows → `past_covariates`, known rows → `known_covariates`). Each group declares version, role, dependencies, availability rule, missing handling and parameters (defaults reproduce the version exactly; non-default parameters are recorded as `feature_params`). Past-only inputs use history ≤ origin, known-future ones only if genuinely known (astronomy, issued forecasts). An entirely empty selected row is refused, not run. Version each group; definitions and actual arrays enter provenance and exploratory cache keys. Select on chronological validation (`weather-lab features select`), test once; the October AROME suite is development data.
- **Readable names in the app.** Every system id in `configs/` needs a readable name and description (`MODELS`) and a colour/line style (`STYLE`) in `src/weather_lab_app/labels.py`; `tests/test_app_packaging.py::test_labels_cover_every_configured_system` enforces it. Keep colours within the system's family (Timestral blues, Timestral + AROME teal, AROME orange, references greys). New experiments shown in the app go in `labels.EXPERIMENTS`. Identifiers ending in `-cal` mean calendar inputs, never calibration; do not name them otherwise in the UI or docs.
- Station ids are 5-character strings. Quote them in YAML.
- Download budget: about 500 MiB for initialization. Report needs that exceed it instead of fetching silently.
