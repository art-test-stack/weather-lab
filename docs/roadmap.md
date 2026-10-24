# Roadmap: subsequent experiments (documented, not implemented)

Each step keeps the frozen suites and adds a **new suite version** whenever dates, sampling or preprocessing change. New systems join through the adapter contract in `src/weather_lab/forecast.py` (`Adapter.predict(CaseBatch) -> ForecastBatch`). Scoring and UI code need no changes.

## 1. Complete the historical benchmark

- Fetch SYNOP 2018–2024 (about 7 MB per year) and run `configs/benchmark.yaml`: daily 00 UTC origins, five stations, baselines fitted on 2018–2022.
- Choose the model and context **on validation 2023 only**: 28M vs 209M (837 MB, add as another `timestral` model entry), and context lengths of 7, 30 and 90 days via `context_days`. Freeze the choice, then score test 2024 once.
- Report temporal block-bootstrap intervals for paired differences. With daily origins, set the block length from the autocorrelation of the paired loss differences. Two days is the floor imposed by the 48 h horizon overlap.
- Keep the smoke results labelled as integration checks.

**Smallest next experiment:** run the validation half of `benchmark.yaml` with 28M at 7/30/90-day context. That needs about 35 MB of SYNOP downloads and a few minutes of CPU.

## 2. Extend the AROME suite

- The public archive keeps only 14 days, so it cannot give multi-year access. Options:
  - **prospective collection**: a daily job running `weather-lab data fetch --config configs/arome_smoke.yaml` with rolling dates, at about 22 MB per run for T2m + MSLP;
  - a longer issued-forecast archive obtained from Météo-France.
- Freeze `arome-issued-v2` spanning at least one year, so it includes seasons and enough warm-tail and windstorm cases. Only runs whose files were available at each origin qualify.
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
