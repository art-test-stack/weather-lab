# What the AROME covariate contains, and why the "missing history" variant ignored it

Audited 2026-10-24 on suite `arome-issued-v2`, which has the same 60 cases as `arome-issued-v1`.

Reproduce:
- `uv run weather-lab diagnose arome-inputs --config configs/arome_v2.yaml --case arome-issued-v2/07630/20261021T06`
- `uv run weather-lab diagnose covariates --config configs/arome_v2.yaml [--case …]`

## 1. Inputs and terms

A **covariate** is an additional input series that the model does not forecast. For AROME-conditioned Timestral the model receives:

1. the station's past observations: the targets `t, pmer, u, v` over 30 days (240 steps), up to and including the origin;
2. calendar rows (sin/cos hour and day of year), known over history and horizon;
3. one AROME row each for `t` and `pmer`. Over the **future** (horizon) segment, this row holds the issued forecast of the run available at the origin. Over the **history** segment, what it holds depends on the variant.

| System (v2 id) | History segment of the AROME row | Future segment |
|---|---|---|
| `timestral-28m-mv-cal` (A) | no AROME row at all | no AROME row |
| `…-arome-futureonly` (B) | **missing values** (NaN) | issued forecast |
| `…-arome-archived` (C) | **archived AROME forecasts** from earlier runs; missing where no qualifying run exists | issued forecast |
| `…-arome-substobs` (= v1 `…-arome`) | **substituted station observations**: not AROME values | issued forecast |

No historical AROME analyses are used anywhere. The v1 variant `timestral-28m-mv-cal-arome` used substituted observations. This was declared in its configuration (`arome: observations`) but labelled as a "stitched best-estimate series". v2 names it explicitly, and missing historical AROME is never replaced with observations unless the `substituted_observations` mode is requested by name.

### Rules for archived historical values (`arome.archived_history`)

A value at history valid time *v*, for a case with origin *o*, comes from the most recent fetched run *r* that satisfies all of the following:

- *r* was in the archive by min(*o*, *v*) (archive `Last-Modified`; see [availability evidence](../availability_evidence.md)). The forecast must therefore have been available before the current origin **and** before its own valid time, as a forecast issued ahead of *v* should be.
- *r* is earlier than *v*.
- *r* holds a value for *v*.

`CaseBatch` re-checks the claimed vintage: run time + lead = valid time, and no value without a source run. Only the twelve 03 UTC runs of the suite were fetched, at leads 6–51 h. Consequently:

- History before 2026-10-10 09Z is missing for every case.
- At 06Z the same-day run is in the archive (05:4x Z), but its lead 3 h was not fetched, so the previous day's run supplies the value at lead 27 h.

## 2. Example: Toulouse-Blagnac (07630), origin 2026-10-21 06 UTC, temperature (°C)

Excerpt from the 256-row table (`experiments/arome_inputs/<case>/table.csv`). The future observed values are for verification only and are not inputs.

| valid time (UTC) | segment | observed | AROME value (archived mode) | AROME run | lead (h) | availability source | v1 input (substituted obs.) |
|---|---|---|---|---|---|---|---|
| 10-10 03:00 | history | 14.3 | missing | – | – | none: no fetched run available before both origin and valid time | 14.3 |
| 10-10 06:00 | history | 14.1 | missing | – | – | none | 14.1 |
| 10-10 09:00 | history | 16.8 | 16.19 | 10-10 03Z | 6 | archive Last-Modified 05:44Z ≤ min(origin, valid time) | 16.8 |
| 10-10 12:00 | history | 19.6 | 20.34 | 10-10 03Z | 9 | same | 19.6 |
| 10-21 00:00 | history | 7.2 | 8.76 | 10-20 03Z | 21 | archive Last-Modified 05:47Z ≤ min(origin, valid time) | 7.2 |
| 10-21 03:00 | history | 7.5 | 5.66 | 10-20 03Z | 24 | same | 7.5 |
| 10-21 06:00 | history | 8.8 | 7.73 | 10-20 03Z | 27 | same (10-21 03Z lead 3 h not fetched) | 8.8 |
| 10-21 09:00 | future | (13.2) | 13.13 | 10-21 03Z | 6 | archive Last-Modified 05:45Z ≤ origin 06:00Z | 13.13 |
| 10-21 12:00 | future | (15.8) | 16.30 | 10-21 03Z | 9 | same | 16.30 |
| 10-22 06:00 | future | (6.3) | 6.23 | 10-21 03Z | 27 | same | 6.23 |
| 10-23 06:00 | future | (15.0) | 13.83 | 10-21 03Z | 51 | same | 13.83 |

## 3. Coverage and the possible lead-time mismatch (all 60 cases)

- **Coverage.** Archived values cover 18% of history steps on average. Coverage rises from 0% (origin 10-10) to 37% (origin 10-21). The remaining history is missing.
- **Lead distribution.** History uses leads 6–27 h (uniform, median 16.5 h). The future segment uses leads 6–51 h (median 28.5 h).

The difference in lead distributions is not leakage: every value obeys the vintage rule. It is still a potential distribution mismatch, because the model sees a short-lead archived series in the history and a longer-lead issued forecast over the horizon. Checked as AROME temperature MAE against observations:

| lead (h) | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 30–51 (future only) |
|---|---|---|---|---|---|---|---|---|---|
| history segment | 0.85 | 1.13 | 0.82 | 0.81 | 1.17 | 1.21 | 1.31 | 1.37 | n/a |
| future segment | 0.80 | 1.02 | 0.89 | 0.83 | 1.06 | 1.11 | 1.19 | 1.27 | 0.95–1.44 |

At matching leads the error levels agree to within about 0.1 °C. The history segment has no examples of 30–51 h errors, which grow to 1.44 °C at 51 h. History counts repeat the same archived values across overlapping case windows, so they are not independent samples. Twelve October days cannot characterise this mismatch beyond that.

## 4. Diagnosis of the "missing history" result (v1 `nanpast`, v2 B)

### Setup

All 60 cases and the Timestral-28M checkpoint are held fixed, along with the observations (asserted byte-identical across variants), the 240-step context, the calendar rows, batch size and upstream defaults (`symmetric_averaging=True`, `clip_nonnegative=False`). The arrays compared are exactly those built by the project adapter (`Timestral.build_inputs`). The perturbation adds +5 °C to the **future segment of the temperature covariate only**; its units are °C.

### Maximum absolute output differences over all 21 quantiles

| comparison | all variables, all cases | temperature, all cases | Toulouse 10-21 case |
|---|---|---|---|
| A vs B (future-only AROME rows added) | 1.50 | 1.38 | 0.22 |
| A vs C (archived history + future AROME) | 30.07 | 10.16 | 5.30 |
| **B vs B + 5 °C** | **1.2 × 10⁻⁴** | **2.1 × 10⁻⁵** | **1.2 × 10⁻⁴** |
| C vs C + 5 °C | 25.80 | 25.80 | 2.95 |
| probe B1 vs B1 + 5 °C (B plus **one** observed historical value at the first context step) | 1.09 | 0.77 | 0.13 |

### Trace through the adapter and the pinned upstream code (`timestral@a534bdb`)

1. **Adapter.** The arrays are passed unchanged: for B, the AROME rows are NaN over the 240 history steps, with 16 real future values in `known_covariates` (B, C, T + H). No project code drops or masks them.
2. **`TimestralForecaster._assemble_rows`.** It splits each known-covariate row into history and future segments and sets NaN to 0 with an observation mask. For B's AROME rows: 0 observed history steps and 16 observed future steps in every case. The future values are passed on as `known_future` with `known_future_mask=True`.
3. **`TimestralModel._predict_window`.** The future values are concatenated into the encoder window. `_known_cov_scaling` normalizes each known row with **whole-row** statistics over its observed positions, so for B those are computed from the future values alone. The values therefore do reach the network input.
4. **Masks (`timestral/masks.py`).** Encoder key validity is `compute_enc_key_validity(obs_per_patch, attend_missing=True)`: "A patch is a valid key if it is observed, or if … the row's series has started at or before it. Leading all-unobserved patches are left-padding and are never valid keys."
   - For B's AROME rows, the measured number of valid context key patches is **0 of 8 in all 60 cases**; only the horizon patch is valid.
   - Target decoder tokens read encoder slots `j < anchor`, which are context patches only (`build_two_stream_time_mask`).
   - Variate attention on encoder slots requires the key row's slot to be valid (`build_two_stream_variate_mask`).
   - The AROME row's own decoder token can read only its valid context keys, and it has none.
   - So no attention path carries the row's future values into the targets in a single decode block. The A vs B difference comes from the rows' presence, through their decoder tokens and normalization, not from their values.
5. **Probe B1.** One observed value at the first history step makes that row "started": its 8 context patches become valid keys, and the same +5 °C perturbation now changes the outputs (up to 1.09).

### Conclusion

- **No implementation issue was found in the project code.** The future AROME values reach the model input. They are ignored because the pinned model's masking treats a covariate row with no observed history as not yet started. This is documented upstream behaviour, observed here for a single decode block (horizon 16 steps, within one 32-step patch).
- This does **not** show that future-only covariates are unsupported in general. In multi-block decodes, upstream feeds known-future values back into the context, which would make the row "started" for later blocks. That case was not tested.
- Practical consequence for this project: a historical segment of the AROME row is needed for AROME to have any effect. The only choices are archived forecasts, honest but sparse here, or substituted observations, which is a different input.
- **Observed instability.** In C, a 5 °C shift of the future covariate moves some outputs by more than 5 °C: up to 25.8 °C, at the 0.99 quantile and the last lead, for origins with 33–37% history coverage. The median moves by up to 6.4 °C. Sensitivity grows with history coverage, and is exactly zero at origin 10-10, which has no coverage. The cause has not been established; it is recorded as a limitation of the conditioned variant.

Because there is no defect in our code, no regression test was added for this behaviour. `tests/test_arome_history.py` checks what this project does control: the vintage rule, no substitution, and the exact rows each mode builds.

## 5. Follow-up (2026-10-25): forecast-only response with reordered and local perturbations, both checkpoints

A uniform shift of the future covariate cannot isolate its use: the pinned code normalizes a known
covariate row by its own mean and standard deviation over observed positions, and with the history missing
those are exactly the 16 future values, so a uniform +5 °C cancels out (normalized change 0). The new check
(`weather-lab diagnose forecast-only`) reverses the 16 future values, or adds +5 °C at the 24 h lead only,
with masks fixed, and verifies that the normalized input changes (3.6–4.7 standard units). Temperature
medians then move by about 0.001–0.002 °C for both Timestral-28M and -209M (max quantile change 0.03 / 0.05
°C), against 0.13–0.84 °C mean and up to 15 °C max in the archived-history control. 209M responds slightly
more than 28M but not meaningfully. Table: [../experiments/conditioning_matrix.md](../experiments/conditioning_matrix.md) §4.
