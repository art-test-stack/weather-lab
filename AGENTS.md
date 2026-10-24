# Agent instructions: Timestral Weather Lab

## Ground rules

- **Frozen manifests.** Never edit or regenerate a manifest in `benchmarks/manifests/` once scores exist. A change to dates, stations, eligibility, preprocessing or leads means a new suite version, with a new `suite`/`manifest` value in a new config. `data prepare` refuses silent changes. Keep it that way.
- **No pooling.** Never rank systems across suites, splits, variables or case sets. Never sum errors across °C, hPa and m/s.
- **Labels.** Smoke and AROME-initialization scores are integration checks. Timestral results are inference with no local fine-tuning. Calibration, fine-tuning and emulation are different things and must be labelled as such. Timestral is not WeatherGenerator.
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
uv run python app.py
```

## Conventions

- Python 3.12 with `uv`. Dependencies are pinned in `uv.lock`. Regenerate `requirements.txt` with `uv export --no-dev --no-hashes --no-emit-project -o requirements.txt` whenever `uv.lock` changes.
- Timestral is pinned to commit `a534bdb` and the checkpoint revisions in `configs/*.yaml` / `forecast.CHECKPOINTS`. Bump them deliberately and record the change in `docs/sources.md`.
- Bulk data (`data/raw`, `data/processed`), `runs/` and weights are git-ignored. `data/examples/` holds the small committed offline replay; refresh it with `weather-lab export-example`.
- Keep scientific logic in `src/weather_lab/`. UI callbacks in `app.py` only read saved results. The exploratory panel never writes to `runs/`.
- Station ids are 5-character strings. Quote them in YAML.
- Download budget: about 500 MiB for initialization. Report needs that exceed it instead of fetching silently.
