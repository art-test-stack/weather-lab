# Offline replay examples (real data, cached results)

Each sub-directory is a saved run copied by `weather-lab export-example`. The app replays these runs when no local `runs/` exist, and labels them "cached example (offline replay)".

- `synop-historical-v1/`: smoke run of the historical suite (96 cases, 2023–2024, Toulouse and Paris-Orly). Integration check.
- `arome-issued-v1/`: AROME comparison run (60 cases, 2026-10-10 … 10-21, 5 stations). Integration study. It includes `arome_station_extract.parquet`, the AROME station values used, so the suite can be re-prepared after the 14-day archive has rolled over.

Every directory holds predictions, targets, scores, the frozen manifest, run provenance (`run.json`), the observation windows the explorer plots, and `ATTRIBUTION.json`, which gives the source URLs, retrieval dates, checksums, processing steps and licences.

Observation and AROME data: Météo-France, Licence Ouverte / Open Licence v2.0 (Etalab). Timestral outputs come from the pinned open-weight checkpoint (Modified MIT licence; see `docs/sources.md`). These are real data. No synthetic values are included.
