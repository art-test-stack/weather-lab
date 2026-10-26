"""Composable forecast configuration: model identity + AROME conditioning + feature groups (with
parameters) + calendar inputs + context + horizon. One representation for exploratory requests and
for the saved benchmark systems (whose recorded `run.json` parameters map into it, `from_system`).

`check` reports why a configuration cannot run, with three distinct kinds:

- "unsupported": the model family cannot take this input (capabilities in `models.FAMILIES`).
- "unavailable": the model could take it, but the data do not exist for this case (no AROME run, no
  archived AROME history, no auxiliary observations, no coordinates).
- "invalid": a value outside what the app allows (horizon, context, empty parameter).

Nothing is ever dropped or substituted: a selected feature that cannot be built is an issue, not a
silent omission, and archived AROME is never replaced by observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from weather_lab import features as feats

TARGETS = ("t", "pmer", "u", "v")
PREDICTION_TARGETS = {"weather": "Direct weather prediction", "arome_error": "AROME + predicted error"}
# conditioning key -> Timestral `arome` mode (benchmark configs / run.json params)
CONDITIONING = {"none": None, "archived": "archived", "substobs": "substituted_observations", "futureonly": "missing"}
_FROM_MODE = {None: "none", "archived": "archived", "substituted_observations": "substobs",
              "observations": "substobs", "missing": "futureonly"}


@dataclass(frozen=True)
class ForecastConfig:
    model: str                                   # registered checkpoint name, e.g. "Timestral-209M"
    conditioning: str = "none"                   # key of CONDITIONING
    features: tuple = ()                         # ((group, ((param, value), ...)), ...) registry order
    calendar: bool = True
    context_days: int = 30
    horizon_h: int = 48
    targets: tuple[str, ...] = TARGETS
    step_hours: int = 3
    prediction_target: str = "weather"

    @classmethod
    def make(cls, model: str, conditioning: str = "none", features=(), **kw) -> "ForecastConfig":
        if conditioning not in CONDITIONING:
            raise ValueError(f"conditioning {conditioning!r}; one of {list(CONDITIONING)}")
        if kw.get("prediction_target", "weather") not in PREDICTION_TARGETS:
            raise ValueError(f"forecast method must be one of {list(PREDICTION_TARGETS)}")
        spec = feats.normalize(features)
        return cls(model, conditioning, tuple((n, tuple(sorted(p.items()))) for n, p in spec.items()), **kw)

    @property
    def feature_spec(self) -> dict[str, dict]:
        return {n: dict(p) for n, p in self.features}

    @property
    def feature_names(self) -> list[str]:
        return [n for n, _ in self.features]

    @property
    def arome_mode(self) -> str | None:
        return CONDITIONING[self.conditioning]

    def forecast_part(self) -> tuple:
        """Everything that determines the predictions except the horizon (a saved forecast can be sliced)."""
        return (self.model, self.conditioning, self.features, self.calendar, self.context_days, self.targets,
                self.step_hours, self.prediction_target)

    def as_dict(self) -> dict:
        return {"model": self.model, "conditioning": self.conditioning, "arome_mode": self.arome_mode,
                "features": feats.definition(self.feature_spec), "calendar": self.calendar,
                "context_days": self.context_days, "horizon_h": self.horizon_h, "targets": list(self.targets),
                "step_hours": self.step_hours, "prediction_target": self.prediction_target}


def from_system(record: dict, horizon_h: int, step_hours: int = 3) -> ForecastConfig | None:
    """A saved system (run.json `systems` entry) as a configuration, from its recorded parameters only.

    None for non-Timestral systems or when a needed parameter was not recorded (nothing is assumed).
    Feature parameters absent from older records are the group defaults of the recorded version.
    """
    from weather_lab.models import CHECKPOINTS

    p = record.get("params") or {}
    if record.get("kind") != "timestral" or "checkpoint" not in p or "context_days" not in p:
        return None
    rev = str(record.get("revision", "")).split("@")[-1]       # recorded as repo@<12-char prefix>
    ck = next((c for c in CHECKPOINTS.values() if c.repo == p["checkpoint"]), None)
    if ck is None or not rev or not ck.revision.startswith(rev):
        return None                               # another revision is another model
    name = ck.name
    versions = p.get("feature_versions", {})
    if any(versions.get(f, 1) != feats.GROUPS[f].version for f in p.get("features", [])):
        return None                               # a different feature version: not this configuration
    spec = {f: (p.get("feature_params") or {}).get(f, {}) for f in p.get("features", [])}
    return ForecastConfig.make(name, _FROM_MODE[p.get("arome")], spec, calendar=bool(p.get("calendar", False)),
                               context_days=int(p["context_days"]), horizon_h=horizon_h,
                               targets=tuple(record.get("variables") or TARGETS), step_hours=step_hours,
                               prediction_target=p.get("prediction_target", "weather"))


# --------------------------------------------------------------------------- capability / availability

@dataclass(frozen=True)
class Issue:
    kind: str          # "unsupported" | "unavailable" | "invalid"
    subject: str       # "model" | "conditioning" | "calendar" | "horizon" | "context" | feature group name
    message: str

    def text(self) -> str:
        return f"{self.subject_label()} — {self.kind}: {self.message}"

    def subject_label(self) -> str:
        if self.subject in feats.GROUPS:
            return feats.GROUPS[self.subject].label
        return {"conditioning": "AROME conditioning", "calendar": "Calendar inputs",
                "prediction_target": "Forecast method"}.get(self.subject,
                                                                                         self.subject.capitalize())


@dataclass(frozen=True)
class Availability:
    """What data exist for one case (cheap metadata, no model, no input building)."""

    arome_issued_h: int = 0             # hours of the issued AROME run after the origin (0: none saved)
    arome_run: str | None = None
    arome_history: bool = False         # some archived AROME forecast qualifies before the origin
    aux: bool = False                   # auxiliary SYNOP observations (dew point, cloud) installed
    station_coords: bool = False
    notes: dict = field(default_factory=dict)   # optional per-dependency explanations


def check(cfg: ForecastConfig, data: Availability | None = None, max_horizon_h: int = 120,
          context_range: tuple[int, int] = (7, 90)) -> list[Issue]:
    """Every reason `cfg` cannot run (empty list: it can). `data=None` checks capabilities only."""
    from weather_lab.models import CHECKPOINTS, FAMILIES

    out: list[Issue] = []
    ck = CHECKPOINTS.get(cfg.model)
    fam = FAMILIES.get(ck.family) if ck else None
    if fam is None:
        return [Issue("unsupported", "model", f"{cfg.model} is not a registered checkpoint")]
    if fam.status != "integrated":
        return [Issue("unsupported", "model", f"{fam.name} is not integrated yet (no adapter)")]
    if ck.family == "t0" and cfg.conditioning == "substobs":
        out.append(Issue("unsupported", "conditioning", "t0 requires archived AROME history or explicitly "
                         "missing history; substituted observations are unsupported"))
    if cfg.conditioning != "none" and not fam.known_covariates:
        out.append(Issue("unsupported", "conditioning", f"{fam.name} takes no known-future covariates"))
    if cfg.calendar and not fam.known_covariates:
        out.append(Issue("unsupported", "calendar", f"{fam.name} takes no known-future covariates"))
    correction = cfg.prediction_target == "arome_error"
    if correction and (ck.family != "timestral" or not fam.known_covariates):
        out.append(Issue("unsupported", "prediction_target", "AROME error prediction requires the Timestral adapter "
                         "and known-future forecast-lead inputs"))
    for name in cfg.feature_names:
        g = feats.GROUPS[name]
        if g.role == "past" and not fam.past_covariates:
            out.append(Issue("unsupported", name, f"{fam.name} takes no past-only covariates"))
        if g.role == "known" and not fam.known_covariates:
            out.append(Issue("unsupported", name, f"{fam.name} takes no known-future covariates"))
        miss = [n.split(":")[1] for n in g.needs if n.startswith("history:") and n.split(":")[1] not in cfg.targets]
        if miss:
            out.append(Issue("unsupported", name, f"needs {', '.join(miss)} among the forecast targets"))
        lags = cfg.feature_spec[name].get("lags_h", ())
        if any(lag % cfg.step_hours for lag in lags):
            out.append(Issue("invalid", name, f"lags must be multiples of {cfg.step_hours} h"))
    step = cfg.step_hours
    if cfg.horizon_h < step or cfg.horizon_h % step or cfg.horizon_h > max_horizon_h:
        out.append(Issue("invalid", "horizon", f"{cfg.horizon_h} h; choose {step}–{max_horizon_h} h in steps "
                                               f"of {step} h"))
    lo, hi = context_range
    if not lo <= cfg.context_days <= hi:
        out.append(Issue("invalid", "context", f"{cfg.context_days} days; choose {lo}–{hi} days"))
    if data is None:
        return out
    if cfg.conditioning != "none" or correction:
        if not data.arome_issued_h:
            out.append(Issue("unavailable", "prediction_target" if correction else "conditioning", data.notes.get(
                "arome", "no saved AROME run for this forecast time")))
        elif cfg.horizon_h > data.arome_issued_h:
            out.append(Issue("unavailable", "horizon", f"the AROME run {(data.arome_run or '')[:16].replace('T', ' ')}"
                                                       f" UTC ends {data.arome_issued_h} h after this forecast time; "
                                                       "AROME inputs are never extended (choose ≤ "
                                                       f"{data.arome_issued_h} h"
                                                       + (")" if correction else ", or no AROME conditioning)")))
    if correction and not data.arome_history:
        out.append(Issue("unavailable", "prediction_target", "AROME error prediction needs observed errors from "
                         "qualifying archived forecasts; no archived error history exists before this origin"))
    need = {"arome_history": (data.arome_history, "no archived AROME forecast was available before this forecast "
                                                  "time (needed even without AROME conditioning)"),
            "aux_history": (data.aux, "no dew point / cloud observations are installed for this experiment"),
            "station_coords": (data.station_coords, "no coordinates recorded for this station")}
    for name in cfg.feature_names:
        for dep in feats.GROUPS[name].needs:
            ok, why = need.get(dep, (True, ""))
            if not ok:
                out.append(Issue("unavailable", name, data.notes.get(dep, why)))
    return out


def describe(cfg: ForecastConfig, conditioning_names: dict | None = None) -> str:
    """Compact readable summary (used for labels of applied results)."""
    cond = (conditioning_names or {}).get(cfg.conditioning, cfg.conditioning)
    if cfg.prediction_target == "arome_error" and cfg.conditioning == "none":
        cond = "no additional AROME covariates"
    fl = ", ".join(_feature_text(n, dict(p)) for n, p in cfg.features) or "no additional features"
    method = f"{PREDICTION_TARGETS[cfg.prediction_target]} · " if cfg.prediction_target != "weather" else ""
    return (f"{cfg.model} · {method}{cond} · {fl} · calendar {'on' if cfg.calendar else 'off'} · context "
            f"{cfg.context_days} d ({cfg.context_days * 24 // cfg.step_hours} steps) · horizon {cfg.horizon_h} h")


def _feature_text(name: str, p: dict) -> str:
    g = feats.GROUPS[name]
    changed = {k: v for k, v in p.items() if v != g.defaults[k]}
    if not changed:
        return g.label.lower()
    return g.label.lower() + " (" + "; ".join(
        f"{k.replace('_h', '').replace('_', ' ')} {', '.join(map(str, v)) if isinstance(v, tuple) else v}"
        for k, v in changed.items()) + ")"
