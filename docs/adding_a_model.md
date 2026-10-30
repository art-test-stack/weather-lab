# Adding a model family

The benchmark is model-agnostic below the adapter. Adding a family should touch only the following, with
no changes to scoring, result storage, caching or app conditionals.

1. `weather_lab.models.FAMILIES` – capabilities (below) and `status`.
2. `weather_lab.models.CHECKPOINTS` – name, repo, full revision, inference files, SHA-256.
3. An `Adapter` subclass (its own module), registration in `weather_lab.evaluate.ADAPTERS` for frozen configs, and construction in `weather_lab.forecast_config.build_adapter` for composable configurations. Use a verified retrieval index when retrieval is supported; storage verification and device lifetime stay with callers.
4. Configs: a new config or suite version listing the new system ids.
5. `src/weather_lab_app/labels.py`: readable name, description and colour family for every new id
   (enforced by `test_labels_cover_every_configured_system`).

## Adapter contract (`weather_lab.forecast.Adapter`)

- **Input** is a `CaseBatch`:
  - `history` (N, V, T) at or before each origin, NaN = missing;
  - known-future covariates: calendar from `future_times`, issued AROME in `arome`;
  - optional archived AROME history with provenance.
  - Targets are never in the batch.
- **Output** is a `ForecastBatch` with `point` (N, V, H) in benchmark units (°C, hPa, m/s).
  - Probabilistic systems add `quantiles` (N, V, H, Q) on `QUANTILE_LEVELS`, the 21 levels 0.01, 0.05,
    0.10 … 0.95, 0.99.
  - `validate()` rejects other grids. A family with different native levels must map them in the adapter
    and record the method in `info["quantile_mapping"]`; levels are never relabelled or invented.
  - Deterministic systems return `output="point"` and get no probabilistic scores.
- **Declarations.** `inputs` lists what the adapter consumes (`history`, `calendar`, `arome`), and `family`
  names its `FAMILIES` entry.
- **Preprocessing** belongs to the adapter: scaling, context trimming, covariate encoding. Do not reuse
  Timestral's choices unless they are right for the family.
- **Caching.** Load weights through `weather_lab.models` (`prefetch` for files; one loaded object per
  revision/device/dtype, held in an LRU like `ModelCache`). Keep request data out of the loaded object.
- **Same cases.** A new family is evaluated on the frozen cases of an existing suite through a config with
  that suite and manifest. The scorer's common valid set then forces every compared system onto the same
  (case, lead) pairs; a family that cannot forecast some cases reduces the common set visibly in the
  availability table, never silently.

`FAMILIES` fields: `past_covariates`, `known_covariates`, `missing_values`, `native_quantiles`,
`max_context`, `max_horizon`, `runtime`, `status` (`integrated` | `planned`). Forecast configuration validation uses them to tell **unsupported input** (the family cannot take an input setup) from **not run** (it could,
but no result is saved) and **not integrated yet** (`status: planned`).

## Existing native families

Timestral and TFC t0-gamma are integrated. Read `forecast.Timestral`, `t0_adapter.T0`,
`rag_adapters.RetrievedTimestral` / `RetrievedT0`, and `forecast_config.build_adapter` before adding another path.
The factory preserves each family's native covariate semantics. `development.predict_cases` owns case
batching, result validation, split metadata and single-case failure isolation. Device allocation and CUDA
release belong outside that worker.

Checkpoint identities, hashes and download sizes are declared in `models.CHECKPOINTS`; package versions
are locked in `uv.lock`. t0 weights exceed the initialization download budget and remain an explicit
prefetch rather than a bootstrap default. Do not change precision, batching, preprocessing or checkpoint
revisions as part of structural refactoring.

Parameterize genuinely shared adapter contracts, retain family-specific covariate and negative tests,
and agree a small real-device check before downloading or running new weights. Scoring, independent
saved-file rescoring and recorded-system mapping must remain covered separately.
