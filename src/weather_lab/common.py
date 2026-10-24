"""Configuration, hashing and provenance helpers shared by the pipeline."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Variables of the station benchmark: SYNOP mnemonic -> (display name, unit).
VARIABLES: dict[str, tuple[str, str]] = {
    "t": ("2 m air temperature", "°C"),
    "pmer": ("mean-sea-level pressure", "hPa"),
    "u": ("10 m wind, eastward component", "m/s"),
    "v": ("10 m wind, northward component", "m/s"),
}

STATION_NAMES = {
    "07630": "Toulouse-Blagnac",
    "07149": "Paris-Orly",
    "07110": "Brest-Guipavas",
    "07190": "Strasbourg-Entzheim",
    "07690": "Nice",
}


def unit_of(variable: str) -> str:
    return VARIABLES[variable][1]


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def suite(self) -> str:
        return self.raw["suite"]

    def dir(self, name: str) -> Path:
        p = Path(self.raw["paths"][name])
        return p if p.is_absolute() else ROOT / p

    @property
    def manifest_path(self) -> Path:
        p = Path(self.raw["manifest"])
        return p if p.is_absolute() else ROOT / p

    @property
    def stations(self) -> list[str]:
        ids = [str(s) for s in self.raw["data"]["stations"]]
        for s in ids:
            if not (len(s) == 5 and s.isdigit()):
                raise ValueError(f"station id {s!r} must be a 5-digit WMO string (quote it in YAML)")
        return ids

    def hash(self) -> str:
        return sha256_text(json.dumps(self.raw, sort_keys=True, default=str))[:12]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=path.resolve())


def ts(value: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_revision() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return rev.strip() + ("+dirty" if dirty.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def environment() -> dict[str, Any]:
    import numpy
    import torch

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
    }
    try:
        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                     text=True).stdout.strip() or platform.processor()
    except FileNotFoundError:
        info["cpu"] = platform.processor()
    return info


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)
        f.write("\n")


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)
