import gzip
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from weather_lab.common import sha256_file
from weather_lab.hourly_observations import SOURCE_COLUMNS

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/hourly-archive-v1/prepare.py"
SPEC = importlib.util.spec_from_file_location("hourly_archive_prepare", SCRIPT)
PREPARE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE_MODULE)
prepare = PREPARE_MODULE.prepare

STATION = {"id": "07110", "national_id": "29075001", "department": "29",
           "lat": 48.4, "lon": -4.5, "name": "Brest"}
PRODUCTS = ["meteofrance_arome_france", "meteofrance_arome_france_hd"]
TIMES = ["2024-12-31T22:00", "2024-12-31T23:00", "2025-01-01T00:00", "2025-01-01T01:00"]


def _write_hourly(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        pd.DataFrame(rows, columns=SOURCE_COLUMNS).to_csv(stream, sep=";", index=False)
    return path


def _hourly_row(time, temperature):
    return ["29075001", "Brest", "48.4", "-4.5", "52", time, temperature, "0",
            "1008.2", "0", "4.0", "0", "270", "0"]


def _historical(product, *, bad=None):
    pressure = [None, None, None, None] if product.endswith("_hd") else [1001, 1002, None, 1004]
    result = {
        "utc_offset_seconds": 0,
        "latitude": 48.5,
        "longitude": -4.4,
        "hourly_units": {"temperature_2m": "°C", "pressure_msl": "hPa"},
        "hourly": {"time": TIMES, "temperature_2m": [10, None, 12, 13], "pressure_msl": pressure},
    }
    if bad == "unit":
        result["hourly_units"]["temperature_2m"] = "K"
    elif bad == "shape":
        result["hourly"]["time"] = TIMES[:-1]
    return result


def _bundle(tmp_path, *, bad=None, wrong_raw_sha=False):
    raw_dir, evidence_dir = tmp_path / "raw", tmp_path / "evidence"
    raw_dir.mkdir()
    evidence_dir.mkdir()
    raw_paths = [
        _write_hourly(raw_dir / "H_29_previous-2020-2024.csv.gz", [
            _hourly_row("2024123122", "8"), _hourly_row("2024123123", "9")]),
        _write_hourly(raw_dir / "H_29_latest-2025-2026.csv.gz", [
            _hourly_row("2025010100", "10"), _hourly_row("2025010101", "11")]),
    ]
    raw_records = [{"path": str(path), "sha256": sha256_file(path), "url": f"https://example/{path.name}"}
                   for path in raw_paths]
    if wrong_raw_sha:
        raw_records[0]["sha256"] = "0" * 64
    (evidence_dir / "raw-downloads.json").write_text(json.dumps(raw_records), encoding="utf-8")

    historical_records = []
    for product in PRODUCTS:
        path = raw_dir / f"{STATION['id']}-{product}.json"
        path.write_text(json.dumps(_historical(product, bad=bad if product == PRODUCTS[0] else None)),
                        encoding="utf-8")
        historical_records.append({"path": str(path), "sha256": sha256_file(path), "station": STATION["id"],
                                   "product": product, "requested_lat": STATION["lat"],
                                   "requested_lon": STATION["lon"], "url": f"https://example/{path.name}"})
    (evidence_dir / "historical-downloads.json").write_text(json.dumps(historical_records), encoding="utf-8")
    protocol = {
        "raw_dir": str(raw_dir), "output_dir": str(tmp_path / "out"),
        "evidence_dir": str(evidence_dir), "start": "2024-12-31T22:00:00Z",
        "end": "2025-01-01T01:00:00Z", "stations": [STATION], "products": PRODUCTS,
    }
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    return protocol_path


def test_prepare_writes_unscored_bundle_with_nulls_and_provenance(tmp_path):
    protocol_path = _bundle(tmp_path)
    out = prepare(protocol_path)

    observations = pd.read_parquet(out / "observations.parquet")
    historical = pd.read_parquet(out / "historical.parquet")
    assert observations.time.tolist() == list(pd.date_range("2024-12-31 22:00", periods=4, freq="h", tz="UTC"))
    assert observations.t.tolist() == [8.0, 9.0, 10.0, 11.0]
    assert observations.station.unique().tolist() == ["07110"]
    assert len(historical) == 16
    assert historical.source_kind.unique().tolist() == ["stitched_historical_forecast"]
    assert not {"run", "available_at"}.intersection(historical.columns)
    hd_pressure = historical.query("product == @PRODUCTS[1] and variable == 'pmer'")
    assert len(hd_pressure) == 4 and hd_pressure.value.isna().all()
    base_temperature = historical.query("product == @PRODUCTS[0] and variable == 't'")
    assert base_temperature.value.isna().sum() == 1

    historical_meta = json.loads((out / "historical.meta.json").read_text(encoding="utf-8"))
    assert len(historical_meta["requests"]) == 2
    assert all(request["sha256"] for request in historical_meta["requests"])
    assert "must not be treated as future covariates" in historical_meta["provenance_limitation"]
    availability = json.loads((out / "availability.json").read_text(encoding="utf-8"))
    hd_count = next(row["count"] for row in availability["historical_finite_counts"]
                    if row["product"] == PRODUCTS[1] and row["variable"] == "pmer")
    assert hd_count == 0
    assert "no validation/test split or scoring claim" in availability["requested_2026_scope"]["claim"]
    assert availability["output_hashes"]["historical.parquet"] == sha256_file(out / "historical.parquet")


@pytest.mark.parametrize("bad,message", [("unit", "unit must be"), ("shape", "time array length")])
def test_prepare_rejects_historical_unit_or_shape_errors(tmp_path, bad, message):
    protocol_path = _bundle(tmp_path, bad=bad)
    with pytest.raises(ValueError, match=message):
        prepare(protocol_path)


def test_prepare_rejects_raw_source_hash_mismatch_before_parsing(tmp_path):
    protocol_path = _bundle(tmp_path, wrong_raw_sha=True)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare(protocol_path)


def test_prepare_rejects_changed_coordinates_and_unsupported_product(tmp_path):
    protocol_path = _bundle(tmp_path)
    protocol = yaml.safe_load(protocol_path.read_text())
    protocol["stations"][0]["lat"] += .1
    protocol_path.write_text(yaml.safe_dump(protocol))
    with pytest.raises(ValueError, match="coordinates differ"):
        prepare(protocol_path)
    protocol["products"] = ["meteofrance_seamless"]
    protocol_path.write_text(yaml.safe_dump(protocol))
    with pytest.raises(ValueError, match="unsupported product"):
        prepare(protocol_path)


def test_prepare_preserves_existing_output_directory(tmp_path):
    protocol_path = _bundle(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    existing = output / "keep.txt"
    existing.write_text("preserve this result")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare(protocol_path)
    assert existing.read_text() == "preserve this result"
