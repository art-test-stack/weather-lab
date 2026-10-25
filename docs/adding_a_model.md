# Adding a model family (next: TFC t0-gamma)

The benchmark is model-agnostic below the adapter. Adding a family should touch only the following, with
no changes to scoring, result storage, caching or app conditionals.

1. `weather_lab.models.FAMILIES` – capabilities (below) and `status`.
2. `weather_lab.models.CHECKPOINTS` – name, repo, full revision, inference files, SHA-256.
3. An `Adapter` subclass (its own module) plus one entry in `weather_lab.evaluate.ADAPTERS`.
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
`max_context`, `max_horizon`, `runtime`, `status` (`integrated` | `planned`). The app's experiment matrix
uses them to tell **unsupported input** (the family cannot take an input setup) from **not run** (it could,
but no result is saved) and **not integrated yet** (`status: planned`).

## TFC t0-gamma: integration points (checked 2026-10-25, to recheck and pin at integration)

Sources: [model card](https://huggingface.co/theforecastingcompany/t0-gamma),
[tfc-t0](https://github.com/theforecastingcompany/tfc-t0), PyPI.

| Item | Value found | To do at integration |
|---|---|---|
| Package | `tfc-t0[torch]>=0.6.0`; latest PyPI 0.6.0 (2026-10-13); Python ≥ 3.10, torch ≥ 2.4 (our 2.13.0 fits); MLX extra for Apple silicon | pin the exact version in `pyproject.toml`/`uv.lock`; regenerate `requirements.txt` |
| Weights | `theforecastingcompany/t0-gamma`, main = `e08aee083778f1f32da4974c0a34276776a35c52`; `config.json` (544 B) + `model.safetensors` (1,022,492,908 B ≈ 975 MiB); Apache-2.0, not gated | add to `CHECKPOINTS` with that full revision and the file's SHA-256. **Exceeds the ~500 MiB initialization budget**: do not add it to bootstrap `weights` or `space_preload` without deciding this explicitly |
| Inputs | `context` (B, V, T); NaN = missing; past covariates enter as extra variates (forecast jointly); known-future covariates `future_covariates` (B, F, T + H), NaN = missing | the AROME setups map directly to `future_covariates`. Check how a future covariate with a missing past is treated before assuming the forecast-only setup is meaningful |
| Limits | horizon ≤ 1024 in one pass (longer: autoregressive); max context not stated | measure; record in `FAMILIES["t0"]` |
| Quantiles | native 21 levels: 0.01, 0.05–0.95 step 0.05, 0.99 (same values as `QUANTILE_LEVELS`); other levels interpolated | request exactly `QUANTILE_LEVELS` and assert they come back native |
| Output | `out.quantiles` (B, H, Q), `out.median` | median as point forecast, as for Timestral |
| Flags | card recommends `trim_leading_missing`, `symmetric_averaging`, `clip_nonnegative`, `snap_integers` | `clip_nonnegative` and `snap_integers` are wrong for °C / hPa / m/s; keep them off and record the choice |

Then: `class T0(Adapter)` with `kind="t0"`, `family="t0"`, `ADAPTERS["t0"] = …`, ids such as
`t0-gamma-mv-cal[-arome-archived]` with labels in a new colour family, and a run of
`configs/arome_v2_matrix.yaml`'s cases through a new config using `evaluate --reuse` for the existing systems.
