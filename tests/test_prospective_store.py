from pathlib import Path
from types import SimpleNamespace

import pytest

from weather_lab.prospective_store import Archive, PREFIX


class Bucket:
    def __init__(self):
        self.files = {}
        self.fail_marker = False

    def batch_bucket_files(self, bucket, *, add):
        for source, destination in add:
            if self.fail_marker and "/records/" in destination:
                raise OSError("interrupted marker upload")
            self.files[destination] = Path(source).read_bytes()

    def download_bucket_files(self, bucket, files, **kwargs):
        for source, destination in files:
            Path(destination).write_bytes(self.files[source])

    def list_bucket_tree(self, bucket, **kwargs):
        return [SimpleNamespace(path=p) for p in self.files]


def test_idempotent_records_conflicts_and_observation_versions(tmp_path):
    archive = Archive(tmp_path)
    with archive.writer():
        ident = archive.put("predictions", "case/model", {"point": 4})
        assert archive.put("predictions", "case/model", {"point": 4}) == ident
        with pytest.raises(ValueError, match="immutable"):
            archive.put("predictions", "case/model", {"point": 5})
        for revision in (1, 2):
            archive.put("observations", f"station/time/revision-{revision}", {"value": revision})
    assert len(archive.records("predictions")) == 1
    assert len(archive.records("observations")) == 2
    (tmp_path / "objects" / (ident + ".json")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        archive.get("predictions", "case/model")


def test_incomplete_remote_write_is_invisible_and_retryable(tmp_path):
    bucket = Bucket()
    archive = Archive(tmp_path / "writer", api=bucket)
    bucket.fail_marker = True
    with archive.writer(), pytest.raises(OSError):
        archive.put("forecasts", "run", {"leads": [1, 2]})
    assert archive.get("forecasts", "run") is None
    reader = Archive(tmp_path / "reader", api=bucket)
    reader.sync()
    assert reader.records("forecasts") == []
    bucket.fail_marker = False
    with archive.writer():
        archive.put("forecasts", "run", {"leads": [1, 2]})
    reader.sync()
    assert reader.get("forecasts", "run") == {"leads": [1, 2]}
    assert any(p.startswith(PREFIX + "/records/") for p in bucket.files)


def test_marker_and_download_corruption_rejected(tmp_path):
    bucket = Bucket()
    writer = Archive(tmp_path / "writer", api=bucket)
    with writer.writer():
        writer.put("forecasts", "first", {"value": 1})
    object_path = next(p for p in bucket.files if "/objects/" in p)
    bucket.files[object_path] = b"changed"
    with pytest.raises(ValueError, match="checksum"):
        Archive(tmp_path / "reader", api=bucket).sync()
