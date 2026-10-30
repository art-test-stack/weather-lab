# Refactor architecture and verification

The refactor branch is based on `ea742e3` and retains the accepted batching and Benchmark-visualization changes. It is isolated on `refactor/repository-consolidation` in `.tmp/codex/repository-refactor/worktree`; the approved integration target is `refactor/huge`.

## Ownership map

- `weather_lab.suite`, `data`, `arome`, and `arome_inputs` own frozen case definitions, observation alignment, vintage-checked AROME history, and issued-run selection. `forecast.hourly_inputs` owns shared delayed hourly context and covariate construction; the frozen three-hourly and sliding hourly protocols remain distinct.
- `forecast_config` validates configuration and constructs native or retrieval adapters. `development.predict_cases` owns batch execution, validation, result splitting, metadata and per-case failure isolation. Device allocation and model lifetime remain with the caller.
- `scoring` owns matched numerical scores and uncertainty calculations. `evaluate` owns frozen-suite execution and independent saved-file rescoring; `development` owns development-run artifact verification; `local_benchmark` owns matrix orchestration, resume, comparison and explicit publication; `rag_store` owns content-identified nested RAG artifacts.
- `weather_lab_app.main` composes startup and the two mounted tabs. `forecast_state` owns explicit Forecast draft/capture/run/apply behavior; `forecast_view` owns Forecast display and export; `benchmark` owns the saved-only Benchmark; `results` owns verified discovery; `plots`, `labels` and `methods` own charts, names and help. The app imports science; science never imports the app.
- Frozen specs, manifests, run artifacts and the three reused single-window BF16 runs are unchanged. No dependency was removed: the audited dependencies still have active science, app, ingestion, deployment or experiment roles.

## Deletion and merge ledger

- Removed `rag_evaluate.predict_cpu`; RAG evaluation and Forecast use `development.predict_cases` with the existing single-case default. Its duplicate loop is gone.
- Removed `rag_adapters.foundation`; native and retrieval adapters are constructed by `forecast_config.build_adapter`, with index verification and device lifetime still owned by callers.
- Moved numerical score functions from `evaluate.py` to `scoring.py`; updated production and reproduction callers. `evaluate.common` and `evaluate.score` remain demonstrated compatibility exports for ignored local historical scripts.
- Moved frozen AROME input construction from `evaluate.py` to `arome_inputs.py`; consolidated hourly context building and issued-run selection while preserving each protocol's vintage policy.
- Removed `forecast.weights_cached` and five unreachable app functions: `forecast_state.setup_choices`, old `benchmark.on_suite`/`bench_controls`, and `results.filter_metrics`/`steps_text`.
- Replaced `weather_lab_app.benchmark` with the mounted saved-comparison implementation; deleted the superseded `saved_benchmark.py` module and unmounted `weather_lab_app.rag` panel. Removed `horizon.lead_ranges`, which served only the retired UI. RAG CLI, science, model adapters and Forecast retrieval remain.
- Removed the duplicate legacy Benchmark event/UI assertions; retained active saved Benchmark state, stale-result, busy-guard, matched-support, saved-load and corruption tests. Replaced GUI-only RAG test coverage with metadata-only CLI status coverage and asserted index restrictions at the development execution boundary.
- No dependencies or scored configuration were removed or rewritten. Historical experiment implementations and ignored local reproduction scripts remain intact.

## Sizes

Python files and lines, before at the approved base `ea742e3`, after the merged refactor and AROME comparison correction. Counts include blank lines and comments.

| Area | Before | After |
|---|---:|---:|
| Scientific production | 48 files / 9,666 lines | 49 / 9,694 |
| App production | 11 / 4,095 | 12 / 3,385 |
| Tests | 46 / 6,324 | 47 / 6,392 |
| Tracked experiments | 16 / 2,285 | 16 / 2,286 |

Tests grew to keep executable contracts independent of optional downloaded examples and to verify the actual saved-artifact boundary. The ignored four local historical scripts (652 lines at baseline) were preserved and are not part of these Git counts.

## Verification

- Full fast offline suite: `PYTHONPATH=src WEATHER_LAB_OFFLINE=1 HF_HUB_OFFLINE=1 python -m pytest -q` — **294 passed, 5 opt-in real-model tests deselected**. Existing pandas day-of-year deprecation warnings remain.
- Independent rescoring of a temporary view of a saved run: CLI reports **byte-for-byte** identity across point, probabilistic, N/A, availability, paired, warm-tail, case and matched-pair score files. Original artifacts were only linked/read; rescoring wrote only to its temporary output.
- Synthetic input and provenance comparisons for frozen, hourly and native/retrieval adapter paths are recorded in `.tmp/codex/repository-refactor/step-02/` through `step-09/`; parity was exact and targets remained separate.
- `weather-lab-app build-space --out .tmp/codex/step-11/space` succeeded: 120 files, no data or weights.
- The populated 39-configuration saved Benchmark was rendered and inspected earlier in the refactor. Final AROME comparison inspection at the reported hourly origin verified two distinct point-only systems, default selection, rendered chart/error traces, fixed issued-run provenance and retained missing values. No new tests were added for this final correction.
- No full-year inference matrix, real-model download, MPS/CUDA/ZeroGPU hardware run, remote Space build, upload, deployment or publication was performed. The remote Space was not deployed.

## Final AROME comparison correction

The first attempted fix put comparison rows in the TSFM cache, whose return path relabelled every row as the primary system. That attempt was superseded. `forecast.arome_comparisons` now uses the existing raw and seven-day, shrinkage-five bias-corrected adapters, separate from TSFM inputs and prediction caches. App state holds each system under its own identifier; the presentation layer excludes superseded saved rows, retains deterministic outputs and exports the actual applied curves. Native and retrieval forecasts retain their own archive and observation-delay policies. Comparisons use one origin-available issued run; missing leads remain missing. No scientific benchmark artifacts or archived cache entries were rewritten.

The final merged checkout at `76c29ed` passed the same complete offline suite: 294 passed, 5 opt-in real-model tests deselected. All eight independently rescored tables remained byte-for-byte identical. CLI startup, dependency boundaries and the generated 120-file Space content passed. No tests were added during final correction or integration.

Merged Gradio startup and its localhost HTTP configuration were verified: Forecast and Benchmark tabs, broader TSFM introduction, and the registered event graph are served successfully.
