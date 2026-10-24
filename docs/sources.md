# Sources consulted

All sources were consulted on **2026-10-24**. Revisions are exact where the source exposes one. Items marked *unverified* could not be confirmed from the source itself.

## Timestral (model used in this project)

| Source | Revision / date | Facts relied on |
|---|---|---|
| https://github.com/mistralai/timestral | `main` @ `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` ("Timestral release", 2026-10-15). Pinned in `pyproject.toml`. | The package handles inference only: `src/timestral/` has no training, optimiser or loss code. `TimestralForecaster.from_pretrained(path_or_repo_id, device, max_context, revision, local_files_only, …)`. `forecast(context (B,V,T), prediction_length, observation_mask, past_covariates (B,C,T), known_covariates (B,C,T+H), …, symmetric_averaging=True, clip_nonnegative=True)` returns `Forecast.quantiles` with shape (B,V,H,21). NaN marks a missing value. Code licence is Apache-2.0. |
| https://huggingface.co/mistralai/Timestral-28M | sha `49b26409ac87b5935539a5d8ffe7f397b720db47` (last modified 2026-10-19) | 28M parameters (27,821,184 counted), 12 layers, patch size 32, context up to 32,768 steps, 21 quantiles from 0.01 to 0.99, `model.safetensors` 111 MB. Default checkpoint here. |
| https://huggingface.co/mistralai/Timestral-5M | sha `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` | 21 MB. Used only by the opt-in real-model test and the exploratory UI. |
| https://huggingface.co/mistralai/Timestral-209M | sha `9874d252ab0aae16f7f7c999f98f98d6ad7d8365` | 837 MB. Configured as an optional comparison and **not downloaded** during initialization. |
| https://huggingface.co/mistralai/Timestral-123M | sha `5a147287addda85f6acd8da3fb8858fe9af7e8f7` | Not used. |
| Weights licence (`LICENSE` in each model repo) | as of the shas above | "Modified MIT". It requires attribution, and **"You are not authorized to exercise any rights under this license if the global consolidated monthly revenue of your company (or that of your employer) exceeds $20 million"**. Check this before any commercial or employer use. |
| https://mistral.ai/news/timestral/ | dated 2026-10-15 | Sizes 5M/28M/123M/209M. Covariates can be past-only, known-future or categorical. Mistral states CPU latency of 10 ms (5M) to 140 ms (209M) on an 8-thread server. **Fine-tuning is offered "through Mistral Forge"**; no local training API is released. |
| https://docs.mistral.ai/models/timestral-1-0-209m | no date shown | "Public Preview", open weights, Modified MIT, contexts up to 32K, quantile outputs. *Unverified:* whether a hosted API endpoint exists, because the fetched page was truncated. Local inference here needs no API key. |

## Météo-France observations

| Source | Revision / date | Facts relied on |
|---|---|---|
| https://www.data.gouv.fr/datasets/archive-synop-omm (API `/api/1/datasets/archive-synop-omm/`) | dataset id `686f8595b351c06a3a790867`, metadata last modified 2026-10-24T06:30Z, licence `lov2` (Licence Ouverte v2.0) | Annual resources `synop_YYYY.csv.gz` (1996–2026) on `meteofrance.s3.sbg.io.cloud.ovh.net/data/OBS/SYNOP/`. URLs are resolved from this metadata at fetch time. |
| `synop_2022/2023/2024.csv.gz` | resources last modified 2026-08-06; sha256 `82eb9a9e…`, `2cdcc76c…`, `96741aa6…`; full hashes are in `benchmarks/manifests/*.json` | Used for the smoke suite and for fitting baselines. |
| `synop_2026.csv.gz` | retrieved 2026-10-24, last observation 2026-10-23T21:00Z, sha256 `9ca79889…` | Supplies history and verifying observations for the AROME suite. The file is updated daily, so the manifest checks the observations the cases actually read, not the file hash. |
| Technical description `OBSERVATIONS_Descriptif_Technique_Données_SYNOP_OMM.pdf` | version 2026-04-20 | Field names and units: `t` in K, `pmer` = mean-sea-level pressure in Pa, `pres` = station pressure in Pa, `ff` = 10-min mean wind in m/s, `dd` = 10-min mean direction in degrees, gusts `raf10`/`rafper` in m/s. |
| Station list `observations-liste-stations-synop-omm-20250710.csv` | uploaded 2026-04-22 | WMO ids (5-character strings), coordinates, altitude. |
| Inspection of the files themselves | — | Cadence is 3-hourly (00…21 UTC) for the selected stations. **There are no quality-flag columns.** A missing value is an empty field. The 2026 file has 18 duplicate (station, time) rows for the 5 stations. Some precipitation fields contain negative values such as −0.1, apparently trace codes (`rr*` is unused here). |

Selected stations: 07630 Toulouse-Blagnac (151 m), 07149 Paris-Orly (89 m), 07110 Brest-Guipavas (94 m), 07190 Strasbourg-Entzheim (150 m), 07690 Nice (2 m). In 2022–2024, `t`, `pmer`, `dd` and `ff` are all at least 99% complete for these stations.

## Météo-France AROME forecasts

| Source | Revision / date | Facts relied on |
|---|---|---|
| https://www.data.gouv.fr/datasets/donnees-pnt-retention-14-jours | id `66d02b7174375550d7b10f3f`, licence `lov2`; directory tree `pnt-tree.json` dated 2026-10-24T16:00Z | The rolling archive keeps 14 days (runs 2026-10-09T18Z to 2026-10-24T12Z at inspection). AROME 0.025° files exist for 00/03/12/15Z runs to +51 h, 09Z runs to +36 h and 21Z runs to +42 h; 06/18Z were empty. Object store `meteofrance-pnt.s3.rbx.io.cloud.ovh.net/pnt/<run>/arome/0025/<package>/…` supports HTTP byte ranges, and its `Last-Modified` header is used as archive-availability evidence. |
| https://www.data.gouv.fr/datasets/paquets-arome-resolution-0-01deg | id `65bd1247a6238f16e864fa80` | Hourly files of about 25 MB (SP1) to 70 MB (HP1). By header scan, the 0.01° SP1/SP2/SP3 packages contain **no MSLP**. Not used. |
| https://www.data.gouv.fr/datasets/paquets-arome-resolution-0-025deg | grid EURW1S40 (55.4N–37.5N, 12W–16E), 6-hour lead groups | The **SP1** package contains 2 m temperature (GRIB2 0/0/0, height 2 m, ~1.0 MB per message) and pressure reduced to MSL (0/3/1, level type 101, ~0.5 MB), plus 10 m u/v. CCSDS packing (template 5.42). Constant fields: `constant-eurw1s40.grib2` (model surface height `h`, land-sea mask `lsm`). |

Measured availability for the 12 runs used (03Z, 2026-10-10 to 2026-10-21): each run was complete in the archive **2.64–2.80 h after initialization** (05:38–05:48Z), before its 06Z origin. The dedicated Météo-France API, which may require an account, was not used.

## WeatherGenerator (future milestone, not integrated)

| Source | Revision / date | Facts |
|---|---|---|
| https://github.com/ecmwf/WeatherGenerator | stable branch **`main` @ `e4c95e77ac3c7f625e813d4b5482e6aff9870641`** (2026-01-20). Default branch `develop` @ `4c98353ceec54dea36ed17ae50dd69890fac521a` (2026-10-20). Tags include `atmo-forecast-small-0.3rc1` → `b6806062…` and `v0.1rc2` → `fc3f3eec…`. | README: "The main branch is the most stable version. If you are running experiments, you should use this branch." Install with uv: `uv sync`, then `uv run train` / `inference` / `evaluate`. Python ≥3.12,<3.13. Apache-2.0. Data are Zarr via `anemoi-datasets` streams (`type: anemoi` for gridded data, `type: obs` for SYNOP and similar). Defaults use flash-attention, FSDP and bf16, so a GPU is needed in practice (our reading of the config; the README does not say so). |
| Checkpoints | — | Not mentioned in the README. `develop` has `scripts/download_experiment.py`, which reads an anonymous S3 bucket `weathergenerator-release` at `object-store.os-api.cci1.ecmwf.int`. Listing shows `checkpoints/atmo/main/af90zz71/models/latest.chkpt` (4.65 GB, 2026-07-29). *Unverified:* which code revision matches that checkpoint (`main` has not moved since January). Not downloaded. |
| AROME support | — | None on `main`. On `develop`, `arome` appears only as an **evaluation region** (37–56N, −12–16E). There is no AROME data stream or regional fine-tuning recipe. CERRA (5.5 km) is the only regional gridded stream. |
| https://weathergenerator.eu/blog/technical-blog-vol-1/ | no date on the page (listed as 2025-05-07) | An Earth-system foundation model: a local "assimilation engine", attention-based global latent, and prediction heads queried at any location and time. Inputs: ERA5, CERRA, SEVIRI, NPP-ATMS, SYNOP. |
| https://weathergenerator.eu/blog/technical-blog-vol-3/ | listed as 2026-05-27 | FastEvaluation metrics: RMSE, MAE, CRPS, SSR, rank histograms, ACC, categorical scores and others. Regrids with earthkit-regrid for model intercomparison. Station verification is not described. |
| https://wgen-fasteval.readthedocs.io/en/latest/ | no version shown | Reads native WeatherGenerator Zarr, and score files with `type: csv` (`metrics_dir/<run_id>/*.csv`, header `,parameter,level,number,score,step,date,domain_name,value`). CSV input only produces score line plots. Station/point ingestion is not documented; non-gridded data must "already be on the same point layout as the model output". **Our station forecasts cannot be fed to its native reader.** At most, our scores could be exported to its CSV format. |

## Hugging Face Spaces (deployment, not done)

| Source | Facts |
|---|---|
| https://huggingface.co/docs/hub/spaces-config-reference | README YAML keys: `sdk`, `sdk_version`, `app_file`, `python_version` (defaults to 3.10). `suggested_storage` is ignored ("persistent storage feature is no longer available"). |
| https://huggingface.co/docs/hub/spaces-sdks-gradio | Dependencies come from `requirements.txt`. |
| https://huggingface.co/docs/hub/spaces-overview | CPU Basic is 2 vCPU / 16 GB / 50 GB non-persistent disk with no hourly cost, **but "creating a Space that runs on compute (Gradio or Docker) requires a paid plan"** (PRO for personal accounts). Free Spaces sleep when idle. |
