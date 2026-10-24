# Offline replay examples (installed, not committed)

The sub-directories hold saved runs. `weather-lab bootstrap` installs them from the immutable snapshot, or recomputes them from official sources. `weather-lab export-example` creates them from local runs. They are git-ignored.

- `synop-historical-v1/`: smoke run of the historical suite. Integration check.
- `arome-issued-v2/`: current AROME comparison, with explicit historical AROME inputs. Integration study under an archive-timestamp assumption.
- `arome-issued-v1/`: earlier version of the AROME comparison, kept unchanged.

The AROME directories include `arome_station_extract.parquet`, so those suites can be re-prepared after the 14-day archive has rolled over.

Every directory holds predictions, targets, scores, the frozen manifest, provenance (`run.json`), observation windows and `ATTRIBUTION.json`. Data: Météo-France, Licence Ouverte v2.0. Timestral outputs: pinned open-weight checkpoint, Modified MIT licence.
