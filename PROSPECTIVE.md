# Prospective evaluation

This workflow collects evidence, freezes predictions, and scores delayed hourly
measurements. The Gradio **Prospective Evaluation** tab reads the saved archive.
It never collects data, loads a checkpoint, performs inference, or writes scores.
Forecast and Benchmark retain their existing event chains and saved results.

## Review and activation

The entry workflow is `.github/workflows/prospective.yml`; its jobs call
`prospective-stage.yml`. **Only `workflow_dispatch` is active.** The proposed
hourly and daily cron entries are commented out. This PR does not deploy the
Space, enable schedules, merge anything, or start an inference service.
New dispatch workflows normally need to exist on the default branch before
GitHub exposes their Run workflow control. Review and merge separately; the CLI
can be invoked explicitly on this branch without registering a workflow.

Before any operational execution, configure GitHub repository secrets:

| Secret | Required permissions / purpose |
| --- | --- |
| `HF_TOKEN` | Read/write access to the existing `ai-testack/weather-lab-storage` bucket. Needs read access to published Benchmark discovery records, pinned source artifacts and retrieval indexes, and write access to `prospective/v1/`. No Space write permission is required. |
| `METEOFRANCE_APPLICATION_ID` | Recommended for unattended hourly observation ingestion: the Basic application credential from Météo-France's API portal, without the `Basic ` prefix. Subscribe the application to Données Publiques Observation. The collector exchanges it for an OAuth access token each run. |
| `METEOFRANCE_TOKEN` | Alternative current OAuth Bearer token for that observation API. Renew before expiry. Takes precedence over the application credential. |

No forecast API credential is needed for the official PNT archive. Missing or
expired observation credentials produce explicit status/gaps; they never select
another source or manufacture hourly SYNOP labels. Use repository secrets, not
workflow inputs, for credentials. Neither authorization headers nor token bodies
are stored in evidence records. The bucket is public: collected weather evidence
and prediction provenance are publicly readable.

All remote writers must run under the **same `prospective-bucket-writer`
concurrency group**, with `cancel-in-progress: false`. The reusable workflow must
only be called through that entry workflow. Local writers additionally hold a
filesystem lock. Do not run a local remote-writing command alongside Actions:
the bucket provides no distributed lock or compare-and-swap operation. The
`--serialized-writer` flag asserts that the caller has obtained exclusive writer
ownership; it does not acquire a cross-machine lock.

## Configuration and model scope

`weather_lab.saved_catalog.saved_configurations` is the unchanged discovery
function shared with Benchmark. Catalog setup copies the existing published
`run.json` / `matrix_setup.json` discovery records, checks their recorded checksum,
and freezes exact saved configurations and checkpoint identities. It never
expands the YAML matrices into new configurations. The installed catalog at
implementation time contained 48 saved configurations.

The frozen catalog includes configuration/feature versions, target type,
conditioning, calendar, context, horizon, precision, retrieval index identities,
checkpoint repository/full revision/SHA-256, and adapter/preprocessing versions.
Reconstructing a different definition is refused. Registry/checkpoint changes
cannot regenerate an old prediction. Existing saved predictions are returned
without inference. A different protocol belongs in a new archive version,
not an overwrite of `prospective/v1`.

Inference uses the existing hourly Benchmark input builder, feature pipeline,
`forecast_config.build_adapter`, model registry/cache and retrieval adapters.
It retains the existing three-hour observation bridge and 48-hour configuration.
No retraining, fitting, new checkpoint, new recipe or exploratory prediction cache
is involved. Raw issued AROME is stored as the deterministic reference.

## Evidence and timing

The official [PNT catalog](https://www.data.gouv.fr/datasets/donnees-pnt-retention-14-jours)
discovers actual issues; the collector does not invent an hourly issuance cycle.
The SP1 0.025° packages contain **hourly leads**, grouped into lead ranges.
Existing GRIB scanning, exact byte-range downloads, station extraction and unit
conversion are reused. The archive retains source subset bytes, SHA-256 hashes,
byte offsets, URL, ETag, Last-Modified, product, issue, valid time, lead, station,
variable and actual collection time. Partial files remain retryable; completed
files are immutable. Native leads 49–51 are retained in addition to 1–48 to support
the existing delayed decision alignment. Shorter issues remain shorter.

One decision is taken on the next UTC hour after the first collected part of
each issue. Inputs are restricted to records actually collected by that decision;
revisions collected later cannot enter history. Forecast history must additionally
have been available before its own valid time. Each model gets its selected
issue, not a stitched future from several issues. A result completed at or after
a target valid time loses that target for prospective scoring, a stricter cutoff
than waiting for observation publication. Predictions record start/end time,
cutoff, exact source identities/input hash, scientific source-file hashes, runtime,
lockfile hash, code revision, configuration, feature report and adapter provenance.

AROME publication latency plus the fixed 48-hour horizon can make a configuration
unavailable: a 51-hour issue first usable at issue+4h cannot cover 48 further hours.
The collector also encounters operational short-horizon cycles. The implementation
does not shorten a saved configuration, extrapolate a forecast, stitch issues or
replace AROME with observations. Raw AROME can have partial lead coverage; models
requiring a complete future record an explicit unavailable case.

The [Météo-France observation API](https://confluence-meteofrance.atlassian.net/wiki/spaces/OpenDataMeteoFrance/pages/853934243)
distinguishes three-hourly SYNOP messages from genuine hourly station measurements.
The real-time collector uses the **hourly station** endpoint with the same stations'
national IDs and preserves their five-character WMO aliases. It retries the last
24 hours, validates returned station/time, converts Kelvin/Pa to °C/hPa, and refuses
to relabel the API's nearest-time response as a requested missing hour. Raw values,
reference/insertion times and source bytes are retained. These real-time values
are raw measurements, not yet the qualified archive's QC-reviewed observations.

Independent preceding-day ingestion retries the previous seven days from the
existing official HOR files. It uses `hourly_observations.read_hourly`, accepts QC
0/1, and leaves absent/rejected values missing. Changed daily extracts are new
versions. At scoring time a later qualified version can replace the earlier raw
measurement; the observation-record identity used for every score remains recorded.
No verification label is interpolated.

Pinned retrieval artifacts must be installed at catalog setup. Their actual
first recorded availability is checked for live inference. Historical error-index
availability assumptions remain visible in the index provenance; the index must
already exist at the live decision, and its existing causal retrieval rule still
requires completed analogue outcomes before the query. Missing indexes or model
weights yield unavailable cases, never a different model.

## Storage, retries and scoring

The archive is in `hf://buckets/ai-testack/weather-lab-storage/prospective/v1/`:

- `objects/<sha256>.json`: content-addressed evidence/prediction/evaluation payloads.
- `records/<kind>/<sha256-of-logical-key>.json`: completion markers pointing to payloads.
- Kinds: `catalog`, `forecasts`, `observations`, `predictions`, `availability`,
  `evaluations`, and `status`.

Payloads upload first, are downloaded and checksum-verified, then their commit
marker uploads last. Local synchronization also publishes markers atomically
after payload validation. A separate catalog completion record prevents partial
configuration uploads from exposing an incomplete model list. Collection and freeze
timestamps retain microseconds to avoid admitting records just after an hourly cutoff.
Partial uploads are invisible and retryable. An identical
logical record is idempotent; different content for an existing prediction key is
an error. Predictions, including terminal unavailable decisions, are never replaced.
A crash between a prediction and its availability entry is repaired on retry.
Observation changes and score recomputations get new records; scoring unchanged
inputs reuses the existing evaluation. Status records preserve attempts and errors.
This is application-enforced immutability on a mutable bucket, not storage-level WORM.

The job graph is collection → inference → scoring. Observation-only runs score
without repeating inference. HTTP requests retry rate limits/transient server
errors, each CLI stage has three attempts, and subsequent polling retries delayed
publication. Commands synchronize durable storage before writing. A hard crash
can lose an inference not yet durably committed; a retry must still pass the real
completion cutoff and cannot backdate its predictions. Terminal missing-input or
model-failure decisions remain unavailable rather than being repaired with later inputs.

Metrics use identical `(cohort, issue, decision/case, station, valid time, variable,
lead)` support across the selected configurations and raw AROME. The scorer saves
MAE, RMSE, signed **G = MAE(AROME) − MAE(model)**, and **|G|** by date, station,
issuance hour and lead. The UI recomputes these descriptive aggregates from saved
predictions/observation versions for the selected common support, including
cumulative horizon selection. Pair counts weight aggregates; |G| is taken after
aggregation. Coverage includes missing observations/predictions, common-support
exclusions and explicitly absent historical issues. It never pools variables or
cohorts, and a performance decrease alone is never presented as proof of leakage.
The archive keeps point predictions for these metrics, with no invented
probabilistic bands or deterministic probabilistic scores.

## Explicit execution

Use `uv run --frozen weather-lab prospective --help`. For local development,
omit `--remote --serialized-writer`; this writes only `data/prospective/v1`.
For remote commands, first ensure no other writer is active, then add both flags.

1. Run workflow mode **setup** with `allow_large_downloads: true` after review.
   It installs the exact saved catalog/indexes, pinned weights, genuine observation
   history and recent hourly API data. The full current model catalog requires
   about **2 GB of checkpoint files plus retrieval artifacts**, exceeding the
   normal 500 MiB initialization budget. The flag explicitly opts in. There is no
   paid GPU service; jobs infer on the GitHub-hosted runner CPU. Actions cache
   holds reusable pinned weights/indexes, never the authoritative evidence archive.
2. Run **cycle** to collect actual published forecasts/recent observations, then
   infer eligible saved configurations and score. The default two-day lookback
   tolerates scheduling delays; decisions not yet reached wait for a later run.
3. Run **observations** to retry preceding-day publication and rescore observation
   revisions. Run **score** to rescore saved artifacts without inference.
4. Backfill in bounded date chunks, with explicit inclusive UTC dates, for example
   mode **backfill**, `start: 2026-10-01T00:00:00Z`,
   `end: 2026-10-07T23:00:00Z`. Setup must already have frozen the catalog.
   Backfill imports the checksum-verified pinned suite's genuine issued runs,
   requests retained official issues, collects genuine historical observations and
   runs explicitly retrospective inference. Acquisition commands have a 500 MiB
   per-invocation data budget; split large ranges instead of silently exceeding it.
   Installing an absent pinned historical source may itself require the explicit
   large-download opt-in.

An equivalent CLI sequence is:

```bash
uv run --frozen weather-lab prospective catalog --allow-large-downloads
uv run --frozen weather-lab prospective models --allow-large-downloads
uv run --frozen weather-lab prospective observations --start 2026-08-01
uv run --frozen weather-lab prospective realtime
uv run --frozen weather-lab prospective forecasts
uv run --frozen weather-lab prospective infer
uv run --frozen weather-lab prospective score
uv run --frozen weather-lab prospective backfill --start 2026-10-01 --end 2026-10-07T23:00Z
uv run --frozen weather-lab prospective infer --start 2026-10-01 --end 2026-10-08T02:00Z --retrospective
uv run --frozen weather-lab prospective score
```

The official PNT retention is 14 days. Earlier October issues only exist if the
verified saved archive actually contains them; absent nominal three-hourly issues
are recorded explicitly in availability. Historical source availability assumptions
are preserved and labelled. October replay is `retrospective-2026-10`; November
backfill is also retrospective. Only genuinely time-frozen November records become
`prospective-2026-11`. Historical publication and observation revision times cannot
be recovered by collecting data later. No old record is relabelled prospective.

The Space reads the bucket mount at `/data/prospective/v1` when present. Else it
reads local `data/prospective/v1`, or `WEATHER_LAB_PROSPECTIVE_ARCHIVE` if set.
Outside a mounted Space, explicitly run `prospective sync --remote` to populate
the local read-only view. Opening/refreshing the tab never synchronizes the bucket.
The initial empty state is expected until collection and inference have run.

## Validation and current limits

```bash
uv run --frozen pytest
HF_HUB_OFFLINE=1 WEATHER_LAB_OFFLINE=1 uv run --frozen pytest -m real_model
```

Tests cover real local batch construction, point-in-time revisions, strict freeze
boundaries, immutable retries, missing inputs, real GRIB parsing/station extraction,
partial horizons, duplicate acquisition, QC and hourly API validation, checksums,
interrupted uploads, pairing, signed/absolute gains, provenance, idempotent scoring,
empty/populated Gradio callback construction, CLI guards and inactive workflow syntax.
Real-adapter checks run the locally cached, pinned Timestral-28M, Timestral-209M
and t0-gamma checkpoints against synthetic **test-only** inputs, never publish scores.
All installed saved configuration definitions were also reconstructed exactly.

Live authenticated observation API responses, bucket writes, GitHub-hosted execution,
and historical acquisition/inference jobs are not exercised by the offline tests.
They require reviewed credentials and an explicit operational run. The complete
historical archive is not guaranteed, and strict publication/horizon/context rules
can leave no shared support for a selected comparison. The simple reader verifies
and scans saved records; long-running archives will eventually need partitioned
indexes rather than a full local mirror. No such index or new dependency is added here.
