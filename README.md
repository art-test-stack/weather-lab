# Weather-lab

Station-level weather forecasting experiments with Timestral, t0-gamma and issued
AROME forecasts, using Météo-France observations. Models run with pretrained
weights and no local fine-tuning.

The Gradio app has two tabs:

- **Forecast:** choose a station, origin, model and inputs, then click **Run forecast**.
  Edits stay pending until Run. Forecasts can predict weather directly or predict
  AROME errors and add them to the issued forecast.
- **Benchmark:** compare saved predictions on matched observations. This tab runs
  no inference. Scores stay separate by variable and case set.

## Run locally

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run weather-lab bootstrap
uv run weather-lab-app
```

Open http://127.0.0.1:7860. The app also runs bootstrap at startup.

Data, weights, generated manifests and saved runs are git-ignored. Bootstrap
installs and verifies the artifacts configured in [configs/bootstrap.yaml](configs/bootstrap.yaml).
It uses a pinned snapshot when configured; official-source mode regenerates the
frozen suites. Older AROME runs require a snapshot once they leave the provider's
14-day archive.

```bash
uv run weather-lab --offline models status
uv run weather-lab models prefetch Timestral-28M Timestral-209M
uv run weather-lab data forecast-hourly
```

The last command explicitly refreshes hourly exploration data. Installing t0-gamma
requires a separate `weather-lab models prefetch t0-gamma` download of about 1 GB,
above the default 500 MiB initialization budget.

## Frozen benchmarks

```bash
uv run weather-lab data fetch --config configs/smoke.yaml
uv run weather-lab data prepare --config configs/smoke.yaml
uv run weather-lab evaluate --config configs/smoke.yaml
uv run weather-lab score --run-dir runs/<suite>/<run-id>
```

Rescoring uses saved predictions without loading models and reports whether score
files match byte-for-byte. `evaluate --reuse <run-dir>` reuses verified identical
systems. Suite definitions live in [benchmarks/specs](benchmarks/specs); changing
dates, stations, inputs or leads requires a new suite version.

See [configs](configs) for experiments and
[docs/experiment_protocol.md](docs/experiment_protocol.md) for scoring and selection rules.

## Local hourly matrix

[configs/hourly_matrix_2026_bf16_v2.yaml](configs/hourly_matrix_2026_bf16_v2.yaml)
defines 39 model/input configurations with 30-day contexts and 48-hour horizons.

```bash
uv run weather-lab local-benchmark plan
uv run weather-lab local-benchmark run --out runs/local-hourly-2026 --device mps
uv run weather-lab local-benchmark compare --out runs/local-hourly-2026
```

Use `cpu`, `mps` or `cuda` for the installed hardware. Repeating the run command
verifies and skips completed setups. Changed code, device or settings require a
new output directory. The FP32 alternative is
`--config configs/hourly_matrix_2026_fp32_v2.yaml`; BF16 and batching can change
numerical results.

Runs save predictions, targets, availability and provenance. The shared comparison
uses identical observed pairs across configurations, AROME and persistence, per
variable. Uploading is a separate action:

```bash
uv run weather-lab local-benchmark publish --out runs/local-hourly-2026
```

Retrieval experiments use immutable observation archives and indexes; see
`uv run weather-lab rag --help` and [configs/rag_weather_hourly_v3.yaml](configs/rag_weather_hourly_v3.yaml).
Exploratory forecasts are cached separately from scored runs.

## Interpretation

- Smoke and October AROME suites are integration/development checks. They do not
  establish held-out accuracy or calibrated uncertainty.
- Model and context selection use validation data. Test results must not guide selection.
- Historical AROME inputs are labelled archived, missing or substituted observations.
  Missing inputs never trigger an unlabelled model or data fallback.
- Inputs respect each forecast origin. Hourly research replays assume observations
  are available three hours after validity and named runs three hours after initialization;
  original publication/revision times are unavailable.
- Deterministic AROME forecasts have no probabilistic intervals. Model quantiles
  are not a claim of calibrated coverage. Never pool scores across variables or suites.

Data sources, checkpoint revisions and model/data licence terms are recorded in
[docs/sources.md](docs/sources.md). Adapter integration is documented in
[docs/adding_a_model.md](docs/adding_a_model.md).

## Deploy to Hugging Face Spaces

```bash
uv run weather-lab snapshot publish --id <new-snapshot-id>
uv run weather-lab-app build-space --out dist/space
```

Snapshot publication uploads and verifies data, then updates its pinned URL and
checksum in `configs/bootstrap.yaml`. Commit that change before deployment.
The Space build generates its launcher and README metadata from
[src/weather_lab_app/space.yaml](src/weather_lab_app/space.yaml).

[The deployment workflow](.github/workflows/deploy-space.yml) runs manually or on
`space-v*` tags. Configure `HF_TOKEN` and `HF_SPACE_ID` for the `huggingface-space`
environment. The workflow verifies the snapshot, builds the Space and replaces
its contents with the generated build. Startup installs the data snapshot;
opening Benchmark does not launch inference or publish results.

The configured storage bucket is public, including published experimental artifacts
and exploratory caches written there. ZeroGPU allocations are limited to explicit
forecast calls; CUDA models are released after each allocation.
