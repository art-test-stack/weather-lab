# Plan: can Timestral improve on AROME at stations? (next milestone, not run)

Status: **plan and data inventory only** (2026-10-25). No adaptation, calibration or fine-tuning has been
run. Every number below is either measured on data in this repository (marked *measured*) or an estimate
(marked *estimate*). The objective is empirical: the answer may be "only at some stations, leads or for
probabilistic scores", or "no".

## 1. Success criteria, fixed before any tuning

| | Definition |
|---|---|
| **Primary** | 2 m temperature, point forecast (median for probabilistic systems), **MAE over leads 3–48 h**, all five stations, on a final AROME-overlap test period that nobody has inspected (§6). Win = the paired MAE difference against **both raw AROME and the best learned AROME correction** (§3) is below zero, with a 95 % block-bootstrap interval excluding zero (2-day blocks, at least 10 blocks: `MIN_BLOCKS_FOR_CI`). |
| Secondary, accuracy | MSLP MAE; RMSE; MAE by station and by lead (3–12, 15–24, 27–48 h), reported descriptively, without a significance claim per cell. |
| Secondary, uncertainty | Pinball loss, 80/90 % coverage and width, against a **probabilistic AROME baseline** (raw AROME + empirical residual quantiles fitted on earlier data, roadmap §0.1). Raw AROME is deterministic and gets no probabilistic score. |
| Not a win | A pinball or coverage gain without lower MAE is reported as "better calibrated uncertainty", never as "beats AROME's accuracy". A gain at one station or lead range is reported as that, not as an overall result. |

Shared cases: one frozen suite, the scorer's common valid set (every compared system on the same (case, lead)
pairs), the same 3-hourly leads and the same issued AROME run per case.

## 2. Development diagnostics (October suite, already inspected)

`arome-issued-v2` (12 origins, 10–21 Oct 2026) has been looked at many times. It is **development data**: use it
for error audits and plumbing tests, never as evidence of an improvement.

Audit, from saved predictions and without new inference (the Benchmark lead-time view and `weather_lab.horizon` already do
most of this):
- error by lead, station, hour of day (origin is always 06 UTC, so valid hour = lead mod 24);
- AROME residual (observation − AROME) by station (*measured*, t, 960 pairs): mean +0.20 / −0.19 / +0.64 /
  +0.14 / −0.26 °C and MAE 0.83 / 0.99 / 1.18 / 1.24 / 1.19 °C for Brest / Paris-Orly / Strasbourg / Toulouse /
  Nice;
- grid mismatch (*measured*): nearest-point distance 0.5–1.1 km; model minus station height −5 to −3 m; **Nice's
  nearest point is sea** (land-sea mask 0, model height 2 m). It is the one station where AROME is not the most
  accurate system (`conditioning_matrix.md`);
- the existing conditioning variants on the same cases (28M/209M × no AROME / archived / observed history /
  forecast only). Season cannot be audited: the suite is a single autumn fortnight.

## 3. Inexpensive learned references (must exist before any Timestral claim)

All fitted only on issued-forecast/observation pairs whose observation was available before each forecast's
origin (rolling, trailing windows):

1. **Bias-corrected AROME**: AROME + mean residual per station × lead group, over the previous N days (N chosen on
   validation from {7, 14, 30, 60}).
2. **Regularised linear local correction / blend**: per station, ridge regression of the observation on AROME at
   that lead, the last observation before the origin, the AROME error at the origin, and hour-of-day terms. The
   penalty and window are chosen on validation.
3. **Probabilistic AROME baseline**: (1) or (2) plus empirical residual quantiles by station × lead group, fitted
   on a separate, earlier window than the point correction.

Timestral results are compared with raw AROME **and** with (1)–(3). Each is labelled **post-processing**, not
fine-tuning.

## 4. Main hypothesis: AROME-anchored residual forecasting

Timestral predicts the **correction to AROME**, not the whole weather signal. This is a hypothesis to test.

- **Target.** r(run, station, valid time) = observation(valid) − AROME(run, valid), for the issued run assigned
  to the case (latest run available at the origin, as now).
- **Residual history (context).** A residual for a past valid time v becomes usable only once its observation is
  available: in replay, v ≤ origin (the rule already used for observation history). Under live operation, add the
  measured SYNOP publication lag. For each past v, use the residual of the latest run that was issued and
  available before v with lead ≥ 3 h, mirroring `arome.archived_history`. Record the run and lead per step as
  provenance; missing runs stay NaN and are never interpolated.
- **Known-future covariates.** The current run's AROME values over the horizon, and calendar features. Optional:
  AROME's lead as a covariate.
- **Back to weather units.** Final quantile q = AROME(lead) + λ · residual quantile q. Adding the same value to
  every quantile at a lead keeps them ordered, so no re-sorting is needed. The point forecast is AROME + λ ·
  residual median.
- **Fallback and shrinkage.** λ ∈ {0, 0.25, 0.5, 0.75, 1}, chosen on validation per variable (and per lead
  group if validation supports it). λ = 0 is exactly raw AROME (the zero-correction fallback) and is always a
  candidate, so the selected system can be "no correction".
- **Adapter.** A new `Adapter` (e.g. `timestral-residual`) in `evaluate.ADAPTERS`, with its own preprocessing,
  quantile grid `QUANTILE_LEVELS` and an explicit record of λ and the residual construction in the run's
  provenance. New system ids get labels in `labels.py`; nothing in scoring or the UI changes.

## 5. Data requirement and inventory

Observation history alone cannot stand in for issued AROME history. Both a residual model and the learned
references need many matched (issue time, valid time, observation) pairs across independent weather episodes.

| Item | Available now | Source / evidence | Enough? |
|---|---|---|---|
| Issued AROME 0.025° SP1 (t2m, MSLP), 03 UTC runs | **12 runs**, 2026-10-10 → 2026-10-21, leads 6–51 h (3–48 h after the 06 UTC origin), 5 stations (*measured*: 1,920 station values) | Météo-France rolling 14-day PNT archive; byte-range subsets in `data/raw/arome` (253 MB, ≈21 MB per run, *measured*); availability from archive `Last-Modified` (`docs/availability_evidence.md`) | **No**: 960 residual pairs per variable over 14 valid days, the same pairs as the test cases |
| Archived AROME history inside a 30-day context | 0–37 % of the window, median 18 % (*measured*, `results.input_history`) | same | No: a residual context would be mostly empty |
| Older issued AROME | none | The public archive keeps 14 days only. Whether Météo-France's API or a data request provides older issued runs is **not verified** | Unknown: ask Météo-France |
| SYNOP observations | 2018–2024 and 2026 to 2026-10-23T21Z, 5 stations, 3-hourly | data.gouv.fr Archive Synop OMM (`docs/sources.md`) | Yes, for any period with AROME |
| Station metadata / grid mismatch | nearest point, distance, heights, land-sea mask per station (*measured*) | AROME constant file | Yes |
| 10 m wind from AROME | not collected | SP1 contains u/v (+≈21 MB per run, roadmap §2) | Needed for a wind target |

**Collection plan** (prospective, the only route verified to work): fetch the 00/03/12/15 UTC runs every day before
the archive rolls over. Record `first_seen_available_at` (`weather-lab arome observe`) at collection time, so
availability is evidenced live and not reconstructed later. Freeze multi-month suites with `weather-lab arome refresh-suite`
or a new config. Storage (*estimate*): ≈21 MB per run for t2m + MSLP subsets → ≈7.7 GB per year for one run per day,
≈31 GB for four; the station extract is a few MB. Until ≥ 90 days are collected, only **development pilots** run,
labelled as such.

## 6. Chronological protocol

- **Development**: October 2026 suite (inspected): plumbing and diagnostics only.
- **Fitting / validation**: collected period, rolling-origin. Settings (N, ridge penalty, λ, context length,
  checkpoint) are chosen on validation only.
- **Final test**: a later AROME-overlap period, untouched, of ≥ 60 days (≥ 30 two-day blocks) (*estimate* of
  what the block bootstrap needs to resolve differences of a few percent). Data, adapters, configs and selection
  are frozen (committed spec + config) before the first test run, and the test is scored once.
- This AROME test is **separate** from the 2024 observation-only test (`synop-historical-v1`), which contains
  no AROME forecasts and cannot evaluate any AROME-based system.

## 7. Direct fine-tuning: audit later, do not assume

Keep four things separate: input engineering (§4), learned post-processing (§3), calibration, and actual weight
updates. Before any weight update:
1. Pinned code (`timestral` @ `a534bdb`) documents standalone inference only; there is no official training recipe.
   The absence of a trainer does not make fine-tuning impossible, but nothing is promised until it is shown.
2. Check gradients flow from Python through the forward pass, write a pinball loss on the 21-quantile head, and
   measure memory (28M on CPU, 209M on a small GPU, 240-step context).
3. Run a tiny verified loop: overfit a handful of batches, save, reload and reproduce the outputs bit for bit, then
   confirm the original checkpoint is unchanged.
4. Check the checkpoint licence for derivatives (`docs/sources.md`); keep original checkpoints and zero-shot
   baselines as permanent references.

## 8. Costs and the first experiment

*Measured* inference cost on this Mac CPU: 209M ≈ 11–12 s per 60 cases (one system, four variables); 28M ≈ 2 s.
A year of daily 06 UTC origins × 5 stations (1,825 cases) is therefore ≈ 6 CPU-minutes per 209M system (*estimate*).
The references (§3) cost seconds. Fine-tuning compute is unknown until §7.2.

**First experiment (development pilot, October suite, labelled as such):**
1. Implement the bias-corrected and ridge references and the residual adapter, with λ fixed at {0, 0.5, 1} (no
   selection on these data).
2. Run them on `arome-issued-v2` through `evaluate --reuse` (the 12 existing systems are not recomputed).
3. Check: no leakage (residual history ≤ origin, provenance recorded), λ = 0 reproduces raw AROME exactly, and
   quantiles stay ordered.

**Go / no-go for collection-scale work:**
- **Go** if the plumbing checks pass and residual-Timestral at λ = 1 is not clearly worse than bias-corrected
  AROME on the development data.
- **No-go** (keep AROME + simple correction) if it is clearly worse at most stations and leads. This is a weak
  signal on 12 days; it decides effort, not results.

Expand to the collected suite only after ≥ 90 days of AROME runs exist, and evaluate on the untouched test once.
