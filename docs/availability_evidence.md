# AROME availability evidence

Checked 2026-10-24. Raw evidence is in `data/evidence/availability_2026-10-24T193410Z.json` and `data/evidence/first_seen.jsonl` (both included in the data snapshot).

Reproduce:
- `uv run weather-lab arome evidence --full-checksum <url> …`
- `uv run weather-lab arome observe`

## What the provider documents

- **Météo-France "Modèles et données de prévision"** (Confluence page 621019138, version 2026-09-30T11:39Z). It describes AROME products, runs and leads. For AROME-PI it states "données disponibles 30 minutes après chaque heure ronde", and for PIAF "disponible 10 minutes après l'horaire du réseau". For deterministic AROME packages it gives **no dissemination or publication times**.
- **`description-technique-paquets-arome.pdf`** (data.gouv.fr resource `789a8822-…`). It lists 8 runs per day (00…21 UTC), leads 0–51 h, the 0.025° groups 00–06 … 49–51 and the package contents. It gives **no availability times**. In the archive, 06 and 18 UTC folders were empty, and 09/21 UTC runs were shorter.
- **Data.gouv.fr dataset "Données PNT – rétention 14 jours".** It states only that packages are kept for 14 days. It does not explain any timestamp.

No authoritative record of original publication time was found.

## What the archive timestamps support

| Check | Result |
|---|---|
| Re-query of all 108 AROME files fetched at 18:20Z, repeated at 19:34Z | HTTP 200 for all; `Last-Modified`, `ETag` and `Content-Length` unchanged. Re-downloaded byte ranges of 2 messages per file match the SHA-256 recorded at fetch time (108/108). |
| Full-file SHA-256 for two sampled files (49H51H group of the 10-10 and 10-21 03Z runs) | Recorded (`42b443a6…`, `0415c64a…`) as a baseline for later re-checks. ETags are multipart (`…-4`), so they are **not** content MD5s. |
| data.gouv.fr catalogue vs object store, for the same AROME resources | Catalogue `last_modified` equals the object's `Last-Modified`. `created_at` (2024-02-03) belongs to a rotating resource slot. The catalogue gives no independent publication time. |
| SYNOP annual files on Météo-France's object store | `Last-Modified` for every year 1996–2025 falls within 2026-08-06 15:29–15:31Z. Single-part ETags equal the content MD5 of our cached 2022–2024 files. On this provider's storage, `Last-Modified` records when an object was **written**, here in a bulk re-upload, not when its data were first published. |

An unchanged `Last-Modified` over 74 minutes does not show that it records first publication. These checks did not observe the opposite case (unchanged checksum with a changed header) for AROME files. The SYNOP store shows that bulk rewrites happen, and nothing documents that AROME objects are never rewritten.

## Conclusion and labels

- **Previous and current AROME results (`arome-issued-v1`, `arome-issued-v2`) are retrospective experiments under an archive-timestamp assumption.** A run counts as available at its origin if every needed object's archive `Last-Modified` time is at or before the origin. The measured gap is 2.64–2.80 h after initialization, about 12–22 min before the 06Z origin. This is not a claim of exact operational availability. The same assumption governs the vintage of archived historical AROME values.
- **For new runs, `weather-lab arome observe` (manual, not scheduled) records `first_seen_available_at`**: the time at which all required messages were successfully retrieved. This is a conservative upper bound backed by evidence.
  - Example: run 2026-10-24T03Z was first seen complete at 19:36:20Z, while its objects' `Last-Modified` is 05:41Z. The bound is loose because the utility ran about 14 h later.
  - A re-run keeps the earlier record.
  - It does not reconstruct historical publication times.
- To make a future suite's availability claims evidence-backed, run `observe` shortly after each expected completion time while collecting, and use `first_seen_available_at` instead of `Last-Modified`. Doing this regularly would be a collection service, which was deliberately not set up.
