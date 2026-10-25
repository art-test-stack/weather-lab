"""Readable names, descriptions and colours for the UI. Exact identifiers stay available in Methods."""

from __future__ import annotations

VARIABLE_NAMES = {
    "t": ("Temperature", "°C", "2 m air temperature"),
    "pmer": ("Sea-level pressure", "hPa", "mean-sea-level pressure"),
    "u": ("Wind (west → east)", "m/s", "10 m eastward wind component"),
    "v": ("Wind (south → north)", "m/s", "10 m northward wind component"),
}
SEASON_NAMES = {"DJF": "Winter (Dec–Feb)", "MAM": "Spring (Mar–May)", "JJA": "Summer (Jun–Aug)",
                "SON": "Autumn (Sep–Nov)", "ALL": "All seasons"}
SPLIT_NAMES = {"validation": "Validation (2023)", "test": "Test"}

# Model identifier -> (display name, one-line description).
MODELS = {
    "persistence": ("Persistence", "Last observation repeated at every lead."),
    "seasonal-naive-24h": ("Same time yesterday", "Repeats the last 24 hours of observations."),
    "climatology": ("Climatology", "Historical distribution for the station, time of day and time of year."),
    "arome-raw": ("AROME", "Météo-France's operational weather model, issued run, nearest grid point."),
    "timestral-28m-t": ("Timestral 28M · temperature only", "Timestral forecasting temperature from its own history."),
    "timestral-28m-mv": ("Timestral 28M · no time inputs",
                         "Temperature, pressure and wind forecast together, without time-of-day/season inputs."),
    "timestral-28m-mv-cal": ("Timestral 28M", "Temperature, pressure and wind forecast together, with "
                                              "time-of-day and time-of-year inputs. No AROME."),
    "timestral-209m-mv-cal": ("Timestral 209M", "As Timestral 28M, with the larger 209M checkpoint."),
    "timestral-28m-mv-cal-arome-archived": (
        "Timestral 28M + AROME", "Also receives the issued AROME forecast, plus archived AROME forecasts over the "
                                 "past (missing where none was available)."),
    "timestral-209m-mv-cal-arome-archived": ("Timestral 209M + AROME", "As Timestral 28M + AROME, 209M checkpoint."),
    "timestral-28m-mv-cal-arome-futureonly": (
        "Timestral 28M + AROME (forecast only)", "Receives the issued AROME forecast but no AROME history; the "
                                                 "pinned model then ignores it (see Methods)."),
    "timestral-28m-mv-cal-arome-substobs": (
        "Timestral 28M + AROME (observed history)", "Issued AROME forecast, with station observations standing "
                                                    "in for past AROME values: a different input, not real AROME "
                                                    "history."),
    "timestral-209m-mv-cal-arome-substobs": (
        "Timestral 209M + AROME (observed history)", "As Timestral 28M + AROME (observed history), 209M checkpoint: "
                                                     "observations stand in for past AROME values."),
    "timestral-209m-mv-cal-arome-futureonly": (
        "Timestral 209M + AROME (forecast only)", "Diagnostic: issued AROME forecast, no AROME history; 209M "
                                                  "checkpoint (see Methods for whether it is used)."),
    "timestral-28m-mv-cal-arome": ("Timestral 28M + AROME (observed history)",
                                   "v1 name of the observed-history variant."),
    "timestral-28m-mv-cal-arome-nanpast": ("Timestral 28M + AROME (forecast only)",
                                           "v1 name of the forecast-only variant."),
    "timestral-28m-mv-cal-ctx07": ("Timestral 28M · 7-day history", "56 three-hourly steps of history."),
    "timestral-28m-mv-cal-ctx30": ("Timestral 28M · 30-day history", "240 three-hourly steps of history."),
    "timestral-28m-mv-cal-ctx90": ("Timestral 28M · 90-day history", "720 three-hourly steps of history (selected)."),
}

# One colour family per kind of system: observations ink, Timestral blues, Timestral+AROME teal,
# AROME orange, simple references greys. Line style separates members of a family.
OBS_COLOR = "#0b0b0b"
STYLE = {
    "arome-raw": ("#eb6834", "solid", 3.0),
    "timestral-28m-mv-cal": ("#2a78d6", "solid", 2.6),
    "timestral-209m-mv-cal": ("#184f95", "dash", 2.6),
    "timestral-28m-mv": ("#6da7ec", "dot", 2.0),
    "timestral-28m-t": ("#6da7ec", "dash", 2.0),
    "timestral-28m-mv-cal-ctx07": ("#9ec5f4", "solid", 2.2),
    "timestral-28m-mv-cal-ctx30": ("#5598e7", "solid", 2.2),
    "timestral-28m-mv-cal-ctx90": ("#184f95", "solid", 2.8),
    "timestral-28m-mv-cal-arome-archived": ("#1baf7a", "solid", 3.0),
    "timestral-209m-mv-cal-arome-archived": ("#0f7a54", "dash", 2.6),
    "timestral-28m-mv-cal-arome-futureonly": ("#7fd1b2", "dot", 2.0),
    "timestral-28m-mv-cal-arome-substobs": ("#7fd1b2", "dash", 2.0),
    "timestral-209m-mv-cal-arome-substobs": ("#2e9c78", "dash", 2.0),
    "timestral-209m-mv-cal-arome-futureonly": ("#2e9c78", "dot", 2.0),
    "timestral-28m-mv-cal-arome": ("#7fd1b2", "dash", 2.0),
    "timestral-28m-mv-cal-arome-nanpast": ("#7fd1b2", "dot", 2.0),
    "persistence": ("#a3a29c", "dash", 1.6),
    "seasonal-naive-24h": ("#a3a29c", "dot", 1.6),
    "climatology": ("#8a8984", "dashdot", 1.6),
}
EXPLORATORY = ("#6250d6", "longdash", 2.2)
# Colour family per model family for ids without an explicit STYLE entry (next family: TFC t0, browns).
FAMILY_STYLE = {"t0-": ("#a0522d", "solid", 2.4)}
# Systems whose inputs differ in kind (not just in model): hatched in comparisons, explained in the summary.
DIFFERENT_INPUT = frozenset({"timestral-28m-mv-cal-arome-substobs", "timestral-209m-mv-cal-arome-substobs",
                             "timestral-28m-mv-cal-arome"})


def model_name(mid: str) -> str:
    if mid.startswith("exploratory"):
        return mid.replace("exploratory ", "Exploratory · ")
    return MODELS.get(mid, (mid, ""))[0]


def model_description(mid: str) -> str:
    if mid.startswith("exploratory"):
        return "Run in this session, outside the frozen benchmark; not saved or scored."
    return MODELS.get(mid, ("", ""))[1]


def model_style(mid: str) -> tuple[str, str, float]:
    if mid.startswith("exploratory"):
        return EXPLORATORY
    family = next((v for k, v in FAMILY_STYLE.items() if mid.startswith(k)), ("#52514e", "solid", 2.0))
    return STYLE.get(mid, family)


CHECKPOINTS = ("28M", "209M")


def with_checkpoint(mid: str, size: str) -> str:
    """The same Timestral variant at another checkpoint size (other systems unchanged)."""
    for s in CHECKPOINTS:
        prefix = f"timestral-{s.lower()}-"
        if mid.startswith(prefix):
            return f"timestral-{size.lower()}-" + mid[len(prefix):]
    return mid


def variable_name(v: str, unit: bool = True) -> str:
    name, u, _ = VARIABLE_NAMES[v]
    return f"{name} ({u})" if unit else name


def unit(v: str) -> str:
    return VARIABLE_NAMES[v][1]


# Experiments shown in the app (benchmark id -> presentation). `headline` drives the overview tiles.
EXPERIMENTS = {
    "arome-issued-v2": {
        "name": "AROME vs Timestral · Oct 2026",
        "status": "Integration study",
        "short": "60 forecasts · 5 stations · 10–21 Oct 2026 · 3–48 h ahead",
        "sample": "this 12-day integration study",
        "scope": "60 forecasts: 5 stations × 12 days (10–21 Oct 2026), leads 3–48 h, issued AROME runs from "
                 "Météo-France's 14-day archive.",
        "caveat": "12 autumn days: not evidence about other seasons or extremes. Run availability rests on "
                  "archive timestamps (retrospective).",
        "headline": ["arome-raw", "timestral-28m-mv-cal", "timestral-28m-mv-cal-arome-archived"],
        "reference": "arome-raw",
        "compare": {"timestral-28m-mv-cal-arome-archived": "timestral-28m-mv-cal"},
        "default_systems": ["arome-raw", "timestral-28m-mv-cal", "timestral-28m-mv-cal-arome-archived"],
        "band": "timestral-28m-mv-cal-arome-archived",
        # Input-setup x checkpoint matrix: ids are base.format(size) + suffix. "main" is the setup shown by
        # default; diagnostic setups are selectable but labelled as such.
        "matrix": {"base": "timestral-{size}-mv-cal", "sizes": ["28M", "209M"], "main": "archived",
                   "setups": {"none": "", "archived": "-arome-archived", "substobs": "-arome-substobs",
                              "futureonly": "-arome-futureonly"}},
        "prob_reference": "climatology",
    },
    "synop-ctxval-2023-v1": {
        "name": "History length · 2023 validation",
        "status": "Validation",
        "short": "1,809 forecasts · 5 stations · daily, 2023 · 3–48 h ahead · 90 days of history selected",
        "scope": "1,809 forecasts: 5 stations × daily origins in 2023, leads 3–48 h. Used to choose how much "
                 "history Timestral reads (90 days selected, rule declared before running).",
        "caveat": "Validation year only; the 2024 test has not been run with this choice.",
        "headline": ["persistence", "climatology", "timestral-28m-mv-cal-ctx07", "timestral-28m-mv-cal-ctx90"],
        "reference": "persistence",
        "compare": {"timestral-28m-mv-cal-ctx90": "timestral-28m-mv-cal-ctx07"},
        "default_systems": ["timestral-28m-mv-cal-ctx07", "timestral-28m-mv-cal-ctx90", "persistence"],
        "band": "timestral-28m-mv-cal-ctx90",
    },
    "synop-historical-v1": {
        "name": "Station history · integration example",
        "status": "Integration check",
        "short": "96 forecasts · Toulouse, Paris-Orly · 1st and 15th of each month, 2023–2024",
        "scope": "96 forecasts: 2 stations (Toulouse, Paris-Orly), 1st and 15th of each month, 2023–2024.",
        "caveat": "Small smoke test of the pipeline; baselines fitted on one year.",
        "headline": ["persistence", "climatology", "timestral-28m-mv-cal", "timestral-209m-mv-cal"],
        "reference": "persistence",
        "compare": {"timestral-209m-mv-cal": "timestral-28m-mv-cal"},
        "default_systems": ["timestral-28m-mv-cal", "timestral-209m-mv-cal", "persistence"],
        "band": "timestral-28m-mv-cal",
    },
    "arome-issued-v1": {
        "name": "AROME vs Timestral · v1 (superseded)",
        "status": "Superseded version",
        "short": "Same 60 forecasts as the current AROME comparison · kept for reference",
        "scope": "Same 60 forecasts as the current AROME comparison; kept unchanged for reference.",
        "caveat": "Its conditioned system used observations in place of past AROME values.",
        "headline": ["arome-raw", "timestral-28m-mv-cal", "timestral-28m-mv-cal-arome"],
        "reference": "arome-raw",
        "default_systems": ["arome-raw", "timestral-28m-mv-cal", "timestral-28m-mv-cal-arome"],
        "band": "timestral-28m-mv-cal-arome",
        "superseded": True,
    },
}
# Input setups of the Timestral x AROME matrix (readable, consistent everywhere in the app).
SETUP_NAMES = {
    "none": ("No AROME input", "Observations and time-of-day / time-of-year inputs only."),
    "archived": ("AROME forecast + archived AROME history",
                 "Issued AROME forecast over the horizon; earlier archived AROME forecasts over the past."),
    "substobs": ("AROME forecast + observed history (substituted)",
                 "Issued AROME forecast; station observations stand in for past AROME values. A different input "
                 "construction, not real AROME history (it does not by itself imply future leakage)."),
    "futureonly": ("AROME forecast only (diagnostic)",
                   "Issued AROME forecast with the past AROME segment missing. Diagnostic: the pinned models barely "
                   "respond to it (see Methods)."),
}
DEFAULT_EXPERIMENT = "arome-issued-v2"
EXPERIMENT_ORDER = ["arome-issued-v2", "synop-ctxval-2023-v1", "synop-historical-v1", "arome-issued-v1"]
