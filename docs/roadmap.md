# Roadmap: subsequent experiments (documented, not implemented)

Each step keeps the frozen suites and adds a **new suite version** whenever dates, sampling or preprocessing change. New systems join through the adapter contract in `src/weather_lab/forecast.py` (`Adapter.predict(CaseBatch) -> ForecastBatch`). Scoring and UI code need no changes.

## 0. Next: does Timestral add information beyond AROME? (agreed next experiment)

Current state (arome-issued-v2, temperature, 960 pairs):
- AROME 1.09 °C.
- Timestral 28M 2.52 °C.
- Timestral 28M + archived AROME history 2.19 °C. That is about 13% better than Timestral alone but about 2× AROME's error, with 81% of observations inside its 80% range.

Two steps, in order, on the same frozen cases (and on a longer AROME suite once one exists):

1. **Probabilistic AROME baseline.** Turn the deterministic AROME forecast into a distribution, using the empirical distribution of past AROME errors by station, variable and lead, fitted only on errors observable before each origin. Score it with pinball loss and 80%/90% coverage next to the Timestral systems. Without it, Timestral's probabilistic scores have no AROME-based reference.
2. **Timestral residual correction.** Forecast the AROME error (observation − AROME) with Timestral, from the history of past AROME errors (archived runs, vintage-checked), and add it back to the issued AROME forecast. Compare against raw AROME and the probabilistic AROME baseline on matching cases. This tests directly whether Timestral improves on AROME, rather than replacing it.

Both are post-processing of AROME. Neither is WeatherGenerator training, and neither involves local fine-tuning of Timestral.

## 1. Complete the historical benchmark

- **Done (2026-10-24): context-length validation on 2023.** 90 days was selected under the pre-declared rule; see `docs/experiments/context_length_2023.md` and `configs/frozen/context-selection-2023.yaml`.
- **Next, smallest:** run `configs/benchmark_test_frozen.yaml` once, on 2024 test, with the frozen 90-day context. This needs SYNOP 2024 (already part of the smoke data) and about 3 CPU minutes per 1,800 cases.
- Run 28M vs 209M on 2023 validation with the frozen context before any further test run. That needs the 837 MB checkpoint.
- Report block-bootstrap intervals for paired differences (implemented; 182 blocks for daily 2023 origins).

## 2. Extend the AROME suite

- The public archive keeps only 14 days, so it cannot give multi-year access. Options:
  - **prospective collection**: a daily job running `weather-lab data fetch --config configs/arome_smoke.yaml` with rolling dates, at about 22 MB per run for T2m + MSLP;
  - a longer issued-forecast archive obtained from Météo-France.
- Freeze a new AROME suite (`arome-issued-r<first>-<last>` via `weather-lab arome refresh-suite`, or a multi-month version) spanning at least one year, so it includes seasons and enough warm-tail and windstorm cases. Only runs whose files were available at each origin qualify. For evidence-backed availability, record `first_seen_available_at` with `weather-lab arome observe` at collection time (docs/availability_evidence.md).
- Add 10 m wind (about +21 MB per run), and consider a land-only grid-point rule. Nice's nearest point is currently sea; a new rule means a new suite version.
- Compare raw AROME, standalone Timestral and AROME-conditioned Timestral on matching cases. Also add simple post-processing baselines (e.g. bias-corrected AROME, EMOS) so that "Timestral as post-processor" is judged against standard station post-processing.
- Label all of this as **post-processing**, not WeatherGenerator training.

## 3. Local adaptation

1. **Calibration first.** Fit a transparent method on validation and evaluate on test: per-lead conformal interval widening, or quantile mapping. Report pinball, coverage and width trade-offs, with caveats about time dependence. The initialization results motivate this: conditioned Timestral is over-confident (80% interval coverage of 0.56–0.67).
2. **Fine-tuning audit.** The public `timestral` package (commit `a534bdb`) contains no training loop, loss or optimiser. Mistral's announced adaptation path is Forge, which is not assumed to be a free local API. Before proposing fine-tuning:
   - check that the forward pass through `TimestralModel` is differentiable from Python;
   - write a pinball loss over its 21-quantile head;
   - measure memory for 28M at a 240-step context on CPU or a small GPU.
3. If that is feasible, run a small PyTorch quantile-loss adaptation (e.g. head-only or LoRA) on 2018–2022, with a frozen-versus-adapted comparison on validation. Check the weights licence before any adaptation.

Post-hoc calibration must never be represented as backbone fine-tuning.

## 4. WeatherGenerator regional experiment

- Audit a pinned stable revision. `main` is at `e4c95e7`, but the public checkpoint `af90zz71` dates from 2026-07, so first establish which revision matches it.
- Document inputs (anemoi Zarr streams), the training recipe, compute (flash-attention, FSDP, bf16, so a GPU is needed in practice) and checkpoint licence and size (4.65 GB).
- Scope a small regional AROME experiment: convert a short AROME 0.025° sample to an anemoi-style Zarr stream, fine-tune or evaluate over the `arome` evaluation region, and define a shared verification protocol (gridded verification against analyses plus station extraction).
- Native Zarr/NetCDF handling and spatial verification belong here.
- Station comparison with this repository needs an explicit extraction adapter that matches the target definitions (2 m temperature, MSLP, nearest point or interpolation). Regional-field verification stays a separate suite.
- Do not claim a working integration until a real model run exists.

## 5. Coherent scenarios (only if needed)

If a use case needs trajectories, for example multi-day extremes, add an explicitly justified dependence model, such as a Gaussian copula over leads and variables fitted on validation residuals, or ensemble copula coupling with AROME-EPS. Evaluate dependence (energy score, variogram score), physical plausibility and extremes. Marginal quantile calibration alone cannot establish those properties.
