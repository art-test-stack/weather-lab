"""Methods, cache and integrity help content from recorded results."""

from __future__ import annotations

from weather_lab_app import zerogpu, forecast_state as forecast, results

from weather_lab.common import ROOT
from weather_lab_app.labels import EXPERIMENTS, MODELS, model_description, model_name


QUESTION = ("How do Timestral forecasts compare with AROME's at French stations, 3–48 h ahead, and does giving "
            "Timestral AROME's forecast change its error?")


def methods_md() -> str:
    exp_rows = "\n".join(
        f"| {EXPERIMENTS[s]['name']} | {EXPERIMENTS[s]['status']} | {EXPERIMENTS[s]['scope']} | "
        f"{EXPERIMENTS[s]['caveat']} | `{s}` |"
        for s in results.RUNS)
    seen = sorted({m for r in results.RUNS.values() for m in r.models}, key=lambda m: list(MODELS).index(m)
                  if m in MODELS else 99)
    sys_rows = "\n".join(f"| {model_name(m)} | `{m}` | {model_description(m)} |" for m in seen)
    return f"""
## What is being tested

{QUESTION}

Saved benchmark **experiments** are frozen sets of forecast cases (stations × forecast times), scored separately
and never pooled. **Hourly exploration** is a separate observation archive: the calendar selects any installed
hourly origin, and only Run forecast computes a prediction. Observations are assumed available at validity;
original publication/revision timestamps are unavailable. Its forecasts are never benchmark scores. Timestral
is used as released (no local training).

| Experiment | Status | Scope | Caveat | Identifier |
|---|---|---|---|---|
{exp_rows}

**AROME availability is an assumption.** An AROME run counts as available at a forecast time if Météo-France's
archive file was already published then, judged from the archive's Last-Modified timestamps after the fact (a
retrospective reconstruction, not a live record). Details: `docs/availability_evidence.md`.

## Systems

| Name in the app | Exact identifier | What it is |
|---|---|---|
{sys_rows}

**Time inputs, not calibration.** "Timestral 28M" receives hour-of-day and time-of-year values as known inputs
(identifiers ending in `-cal`, for *calendar*). No system is calibrated or trained locally: Timestral runs as
released. **AROME history**: Timestral only uses an AROME input if it also sees that input's past; "archived
history" uses earlier AROME runs available at the time, "observed history" substitutes station observations (a
different input). Details: `docs/diagnostics/arome_covariates.md`.

## Scores

- **Mean absolute error / RMSE**: average size of the error of the single-value forecast (the median for
  Timestral), in the variable's unit.
- **Pinball loss**: accuracy of the whole predicted distribution (21 quantiles), in the variable's unit; lower is
  better. Only for probabilistic systems.
- **80% range coverage**: share of observations inside the predicted 80% range; ideally 80%.
- All systems in a table are scored on the same forecast–observation pairs.
"""


def cache_md() -> str:
    """Checkpoint files on disk, models loaded in this process, saved predictions, exploratory cache."""
    from weather_lab.status import cache_status

    from weather_lab.forecast import STATS as fstats

    st = cache_status()
    ck = "\n".join(f"| {r['checkpoint']} | `{r['repo']}@{r['revision']}` | {'on disk' if r['on_disk'] else 'absent'} "
                   f"| {r['mb']} MB |" for r in st["checkpoints"])
    loaded = ", ".join(f"{r['repo'].split('/')[-1]} on {r['device']} (~{r['approx_mb']} MB)" for r in st["loaded"]) \
        or "none (models load only for explicit forecast inference or new development benchmark runs)"
    runs = "\n".join(f"| {EXPERIMENTS.get(s, {}).get('name', s)} | `{r.run['run_id'][:60]}` | {len(r.models)} | "
                     f"{r.source} |" for s, r in results.RUNS.items() if not r.run.get("exploration"))
    pc = st["prediction_cache"]
    ms = st["model_stats"]
    return f"""
**Checkpoint files** (Hub cache `{st['hub_cache']}`; offline mode: {st['offline']}). Every input setup of a
checkpoint shares these files.

| Checkpoint | Pinned revision | Files | Size |
|---|---|---|---|
{ck}

**Loaded in this process:** {loaded}. Loads so far: {ms['loads']}, reuses: {ms['load_hits']}, evictions:
{ms['evictions']}, downloads: {ms['downloads']}.

**Saved benchmark predictions** (shown without loading any model):

| Experiment | Run | Systems | Source |
|---|---|---|---|
{runs}

**This process:** Run forecast clicks {forecast.COUNTERS['runs']}, forecasts applied {forecast.COUNTERS['applied']}, exploratory
inputs built {fstats['inputs_built']}, model runs {fstats['inference']}. Editing a setting changes none of these.

**Exploratory prediction cache:** {pc['entries']} entries ({pc['mb']} MB) at `{pc['path']}`: {pc['persistence']}.
Hits {pc['hits']}, misses {pc['misses']} in this process. Entries are keyed by checkpoint revision, input data
and masks, input setup, history length, horizon, quantiles, device and dtype; never reused across revisions or
setups. What survives restarts and rebuilds: `docs/caching.md`.
"""


def integrity_md() -> str:
    rows = "\n".join(
        f"| {EXPERIMENTS[s]['name']} | `{r.run['run_id']}` | {r.source} | `{r.manifest['protocol_sha256']}` | "
        f"{r.manifest['n_eligible']}/{r.manifest['n_cases']} | `{r.run['code_revision'][:12]}` |"
        for s, r in results.RUNS.items() if not r.run.get("exploration"))
    return f"""| Experiment | Run | Source | Protocol hash | Eligible cases | Code |
|---|---|---|---|---|---|
{rows}

The *AROME vs Timestral · v1* experiment is a superseded version kept unchanged; enable "show superseded
versions" in the Forecast tab's advanced panel to browse it. Fitting/adaptation status filters, availability,
paired differences and timings are in the Benchmark tab's advanced section."""


def protocol_md() -> str:
    p = ROOT / "docs" / "experiment_protocol.md"
    return p.read_text() if p.exists() else "See docs/experiment_protocol.md."
