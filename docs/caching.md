# Caching: checkpoint files, loaded models, predictions

Three layers are cached separately. None of them is tracked by Git.

| Layer | What | Where | Code |
|---|---|---|---|
| Checkpoint files | `config.json` + `model.safetensors` of a pinned full revision; SHA-256 verified | Hugging Face Hub cache: `HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub` | `weather_lab.models.prefetch` |
| Loaded models | One `TimestralForecaster` per (repo, revision, device, dtype), lazy, LRU | process memory | `weather_lab.models.ModelCache` (`MODELS`) |
| Benchmark predictions and scores | Saved runs | `runs/<suite>/<run>/`, `data/examples/<suite>/` (snapshot) | `evaluate`, `bootstrap` |
| Exploratory predictions | One parquet per content key | `WEATHER_LAB_PREDICTION_CACHE`, else the configured bucket mount, else `data/cache/predictions` | `weather_lab.predcache` |

## Checkpoint files

- **Registry.** `weather_lab.models.CHECKPOINTS` maps a name (`Timestral-28M`, or just `209M`) to a repo, a
  **full 40-character revision**, the inference file list and the pinned SHA-256 of `model.safetensors`.
  `resolve()` rejects moving references such as `main`. Each run records `weights_path` and the revision.
- **Downloads.** `prefetch` calls `huggingface_hub.snapshot_download(repo, revision,
  allow_patterns=[config.json, model.safetensors])`: only the inference files, nothing else in the repo. The
  Hub cache's own file locks make concurrent downloads safe, and incomplete files stay as `.incomplete`
  blobs, never in the snapshot folder.
- **Verification.** After download, and once per process, sizes and the SHA-256 are checked. A mismatch or an
  empty file raises `CheckpointError`, naming the folder to delete and the command to rerun.
- **Shared files.** Every input setup of a checkpoint (no AROME, archived, observed history, forecast only)
  uses the same revision, so the same files. There is never a second download per variant.
- **Loading.** The model is always loaded from the **local snapshot directory**
  (`TimestralForecaster.from_pretrained(<local dir>)`), so loading itself makes no Hub request.
- **Offline mode.** `WEATHER_LAB_OFFLINE=1`, `HF_HUB_OFFLINE=1` or `weather-lab --offline …`. No Hub request
  is made; a checkpoint that is not on disk fails with `run weather-lab models prefetch <name>`.
- **Paths.** `HF_HOME` and `HF_HUB_CACHE` are honoured consistently, because bootstrap, CLI, evaluation and
  the app all go through `huggingface_hub`. Tokens are read by `huggingface_hub` from its usual locations
  and never logged; the Timestral repos need none.

```bash
uv run weather-lab models prefetch                 # 28M and 209M (default), or e.g. `models prefetch 5M`
uv run weather-lab models status                   # files on disk, loaded models, saved runs, exploratory cache
uv run weather-lab --offline evaluate --config …   # no Hub access at all
```

## Loaded models

- **Reuse.** `MODELS.get(checkpoint, device, dtype)` returns the same object for every adapter and callback
  in the process. The matrix run used one 209M load for both new variants.
- **Retention.** Bounded by `WEATHER_LAB_MAX_LOADED_MODELS` (default 2) and `WEATHER_LAB_MODEL_RAM_MB`
  (default 4096; float32 parameter bytes). The least recently used model is evicted. Nothing loads at
  startup.
- **Request state.** Context, covariates, masks and normalization statistics are built per call
  (`Timestral.build_inputs`) and passed to `forecast()`. The cached object holds only weights. A per-model
  lock serializes inference on the shared module.
- **ZeroGPU.** The GPU is attached only during the `@spaces.GPU` function. The app loads the CUDA copy inside
  it and releases it afterwards (`MODELS.release("cuda")`); GPU-resident state is never assumed to survive
  between calls. Weight files come from the disk cache each time.

## Predictions

- **Benchmark views read saved files only.** Opening Overview or Benchmark never loads weights
  (`tests/test_app_packaging.py::test_benchmark_views_never_load_a_model`; also checked on real runs: the
  model-load counter stayed at 0 while browsing).
- **Exploratory cache key** (`weather_lab.predcache.make_key`): checkpoint repo + full revision, adapter
  version and preprocessing version, SHA-256 of the exact context and covariate arrays (values and NaN
  masks), station, origin, horizon, step, history length, calendar flag, AROME input setup and run, quantile
  levels, device, dtype, torch version, and "deterministic" as the stochastic setting. With context-enrichment
  features, also their definitions (name, version, role, rows) and the SHA-256 of the actual past-covariate
  arrays (values and masks). A different revision, input setup or feature definition never hits.
- **Writes.** The parquet is written to a temporary file in the target directory and renamed
  (`os.replace`); its metadata JSON appears first, so an existing parquet means a complete entry.
- **Concurrency.** Identical concurrent requests compute once: an in-process lock plus a file lock on local
  disk. Locks on FUSE bucket mounts are not relied upon.
- **On ZeroGPU** the cache is checked on CPU before a GPU is requested, so a hit uses no GPU quota.
- **Separation.** Exploratory predictions are never written to `runs/` and never scored.

## Hugging Face Space

Checked on 2026-10-25 against the HF docs ([Spaces storage](https://huggingface.co/docs/hub/spaces-storage),
[config reference](https://huggingface.co/docs/hub/spaces-config-reference),
[manage Spaces](https://huggingface.co/docs/huggingface_hub/main/guides/manage-spaces),
[local cache](https://huggingface.co/docs/hub/local-cache)) and the live configuration of
`ai-testack/weather-lab` (read-only API).

- **Current Space.** Hardware `zero-a10g`. One volume: bucket `ai-testack/weather-lab-storage` mounted
  **read-write at `/data`** (`space_info().runtime.volumes`; `get_space_runtime().volumes` returns `None` in
  huggingface_hub 2.3.0). No legacy persistent storage: the add-on is discontinued and `suggested_storage` is
  ignored. No model volumes, and no Space variables or secrets.
- **The container disk is ephemeral.** `/data` is persistent only because it is a Storage Bucket, not
  because of the path; the docs use `/data` only as an example mount path.

Chosen mechanism:

1. **Weights.** The generated Space README carries
   `preload_from_hub: ["<repo> config.json,model.safetensors <full revision>"]` for Timestral-28M and -209M
   (from `bootstrap.space_preload` and `models.CHECKPOINTS`).
   - Spaces downloads them at build time into the default cache `~/.cache/huggingface/hub`. Preloading
     ignores `HF_HOME`, so the Space sets neither `HF_HOME` nor `HF_HUB_CACHE`, and the runtime loader
     looks in the same place.
   - Startup bootstrap then finds 28M already verified, and an exploratory 209M run needs no download.
   - The Hub cache is not put on the bucket mount: it relies on symlinks and `.locks`, which are unverified
     on the FUSE mount.
2. **Exploratory predictions** go to the bucket: `bootstrap.caches.predictions` →
   `/data/cache/predictions`, used only when the mount exists and is writable.
   - **The bucket is public**, so cached exploratory forecasts are publicly readable. They are derived from
     public data and contain no user input beyond the chosen settings.
   - Set `WEATHER_LAB_PREDICTION_CACHE=off`, or another path, as a Space variable to change this.
3. **Alternative, not used.** Read-only model volumes (`Volume(type="model", source=…, mount_path=…,
   revision=…)` via `HfApi.set_space_volumes`) can only be configured through the API, not the README.
   - Whether `revision` is honoured on Spaces is unverified.
   - A mount is a plain file tree, so the loader would need the mount path instead of the Hub cache.
   - `set_space_volumes` **replaces** all volumes; the existing bucket mount must be included.

What survives what (per the docs and the mechanism above):

| Item | Process restart | Container replacement | Rebuild (new commit, `factory_reboot`) | Evidence |
|---|---|---|---|---|
| Preloaded weights (build-time image layer) | yes | yes (same build) | re-downloaded at build | docs; generated front matter tested; **Space build not yet run with it** |
| Weights downloaded at runtime (not preloaded) | yes | no | no | ephemeral disk |
| Loaded models (memory) | no | no | no | – |
| Benchmark results | re-installed from the bucket snapshot at startup (seconds) | same | same | snapshot mode verified earlier |
| Exploratory cache on the bucket | yes | yes | yes | docs say buckets outlive the Space; **not verified on the Space** |
| Exploratory cache on local disk (`data/cache/predictions`) | yes | no | no | ephemeral disk |

Verified locally on 2026-10-25 (commands and output in the change report):

- one 111 MB download into an empty `HF_HOME`, then one load reused by four input setups;
- an offline process with sockets blocked loaded from that cache, with identical outputs;
- exploratory repeats hit, and a new process reused the entries (restart);
- changing the setup or the revision missed.

**Not yet verified on the Space:** a build with `preload_from_hub`, and cache entries surviving a Space
restart. Both need the deploy workflow run with the new code and a new data snapshot (see the README,
"Publishing new data").
