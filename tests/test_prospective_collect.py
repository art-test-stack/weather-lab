import gzip

import eccodes
import pandas as pd
import requests

from weather_lab import prospective_collect as collect
from weather_lab.prospective_store import Archive


def grib(leads):
    result = bytearray()
    for lead in leads:
        for field in ("2t", "msl"):
            handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            try:
                for key, value in {"Ni": 3, "Nj": 3, "latitudeOfFirstGridPointInDegrees": 54,
                                   "latitudeOfLastGridPointInDegrees": 38,
                                   "longitudeOfFirstGridPointInDegrees": 352,
                                   "longitudeOfLastGridPointInDegrees": 12,
                                   "iDirectionIncrementInDegrees": 10, "jDirectionIncrementInDegrees": 8,
                                   "dataDate": 20261102, "dataTime": 0, "shortName": field,
                                   "forecastTime": lead}.items():
                    eccodes.codes_set(handle, key, value)
                if field == "msl":
                    eccodes.codes_set(handle, "parameterNumber", 1)
                    eccodes.codes_set(handle, "scaledValueOfFirstFixedSurface", 0)
                eccodes.codes_set_values(handle, [283.15 if field == "2t" else 100000.] * 9)
                result.extend(eccodes.codes_get_message(handle))
            finally:
                eccodes.codes_release(handle)
    return bytes(result)


class Response:
    def __init__(self, data, status=200):
        self.content, self.status_code = data, status
        self.headers = {"Content-Length": str(len(data)), "ETag": '"one"',
                        "Last-Modified": "Mon, 02 Nov 2026 03:00:00 GMT"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("not published", response=self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, *args):
        yield self.content


class Session:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def head(self, url, **kwargs):
        self.calls.append(url)
        return Response(self.payload, 200 if "00H06H" in url else 404)

    def get(self, url, headers=None, **kwargs):
        if headers:
            lo, hi = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            return Response(self.payload[lo:hi + 1], 206)
        return Response(self.payload)


def test_real_grib_collection_idempotency_missing_and_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "now_utc", lambda: "2026-11-03T12:00:00Z")
    tree = {"2026-11-02T00:00:00Z": {"arome": {"0025": {"SP1": ["file"]}}}}
    archive = Archive(tmp_path)
    session = Session(grib(range(1, 7)))
    with archive.writer():
        collect.collect_forecasts(archive, "2026-11-02", "2026-11-03", tree=tree, session=session)
        collect.collect_forecasts(archive, "2026-11-02", "2026-11-03", tree=tree, session=session)
    records = archive.records("forecasts")
    assert len(records) == 1
    rows = pd.DataFrame(records[0]["value"]["rows"])
    assert len(rows) == 24  # two stations, two fields, six hourly leads
    assert set(rows.lead_h) == set(range(1, 7))
    assert abs(rows[rows.variable.eq("t")].value.mean() - 10) < .001
    assert sum("00H06H" in url for url in session.calls) == 1
    assert archive.records("status")[-1]["value"]["status"] == "partial"
    other = Archive(tmp_path / "partial")
    with other.writer():
        collect.collect_forecasts(other, "2026-11-02", "2026-11-03", tree=tree,
                                  session=Session(grib([1, 3])))
    assert not other.records("forecasts")[0]["value"]["complete"]


def test_hourly_measurements_revisions_qc_and_delayed_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "stations", lambda: [{"id": "07149", "national_id": "91027002", "department": "91"}])
    stamp = ["2026-11-03T12:00:00Z"]
    monkeypatch.setattr(collect, "now_utc", lambda: stamp[0])
    header = "NUM_POSTE;NOM_USUEL;LAT;LON;ALTI;AAAAMMJJHH;T;QT;PMER;QPMER;FF;QFF;DD;QDD\n"
    def payload(temperature):
        return gzip.compress((header + f"91027002;ORLY;48.7;2.3;80;2026110201;{temperature};1;1000;9;2;1;180;1\n").encode())
    archive = Archive(tmp_path)
    with archive.writer():
        collect.collect_observations(archive, "2026-11-02", "2026-11-02T23:00Z", session=Session(payload(10)))
        collect.collect_observations(archive, "2026-11-02", "2026-11-02T23:00Z", session=Session(payload(10)))
        assert len(archive.records("observations")) == 1
        stamp[0] = "2026-11-03T13:00:00Z"
        collect.collect_observations(archive, "2026-11-02", "2026-11-02T23:00Z", session=Session(payload(11)))
    records = sorted([r["value"] for r in archive.records("observations")], key=lambda r: r["collected_at"])
    assert len(records) == 2
    assert records[0]["rows"][0]["t"] == 10
    assert records[1]["rows"][0]["t"] == 11
    assert records[0]["rows"][0]["pmer"] is None  # QC 9, never interpolated
    assert len(records[0]["rows"]) == 1  # delayed missing hours are not labels
