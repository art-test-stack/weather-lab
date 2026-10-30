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
    # context-enrichment pilot (features-*-v1): same checkpoint/context/preprocessing, one feature group added
    "arome-biascorr": ("AROME, bias-corrected", "Issued AROME plus a shrunk mean of its recent errors at the same "
                       "run lead (only errors observable at the forecast time); deterministic post-processing."),
    "timestral-209m-mv-cal-ctx30": ("Timestral 209M · 30-day history", "240 three-hourly steps; base of the feature "
                                    "validation."),
    "timestral-209m-mv-cal-arome-archived-f-resid": ("Timestral 209M + AROME + recent AROME errors",
        "As Timestral 209M + AROME (archived history), plus the history of AROME's errors at this station (observation − archived AROME, with the forecast lead), only where the observation existed at the forecast time. Inference only, no training."),
    "timestral-209m-mv-cal-arome-archived-f-atmos": ("Timestral 209M + AROME + dew point + cloud",
        "As Timestral 209M + AROME (archived history), plus observed dew point and total cloud cover (history only). Inference only, no training."),
    "timestral-209m-mv-cal-ctx30-f-atmos": ("Timestral 209M · 30-day + dew point + cloud",
        "As Timestral 209M · 30-day history, plus observed dew point and total cloud cover (history only). Inference only, no training."),
    "timestral-209m-mv-cal-arome-archived-f-solar": ("Timestral 209M + AROME + solar elevation",
        "As Timestral 209M + AROME (archived history), plus the sun's elevation at the station (astronomical, known in advance). Inference only, no training."),
    "timestral-209m-mv-cal-ctx30-f-solar": ("Timestral 209M · 30-day + solar elevation",
        "As Timestral 209M · 30-day history, plus the sun's elevation at the station (astronomical, known in advance). Inference only, no training."),
    "timestral-209m-mv-cal-arome-archived-f-tend": ("Timestral 209M + AROME + tendencies",
        "As Timestral 209M + AROME (archived history), plus 3 h and 24 h tendencies of observed temperature and pressure. Inference only, no training."),
    "timestral-209m-mv-cal-ctx30-f-tend": ("Timestral 209M · 30-day + tendencies",
        "As Timestral 209M · 30-day history, plus 3 h and 24 h tendencies of observed temperature and pressure. Inference only, no training."),
}

# RAG pilot systems live in their separate research tab and artifact namespace.
for _mid, _name, _description, _colour in [
    ("analogue-level", "Weather analogues · levels", "Weighted historical continuations retrieved by weather levels.", "#718096"),
    ("analogue-shape", "Weather analogues · shape", "Endpoint-aligned continuations retrieved by recent trajectory shape.", "#4a5568"),
    ("timestral-209m", "Timestral 209M", "Pinned Timestral with observation history and calendar inputs.", "#184f95"),
    ("t0-gamma", "t0-gamma", "Pinned t0-gamma with observation history and calendar inputs.", "#805ad5"),
]:
    MODELS[_mid] = (_name, _description)

# One colour family per kind of system: observations ink, Timestral blues, Timestral+AROME teal,
# AROME error correction purple, AROME orange, simple references greys.
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
    "timestral-209m-mv-cal-ctx30": ("#184f95", "dash", 2.4),
    "timestral-209m-mv-cal-ctx30-f-atmos": ("#5598e7", "dot", 2.0),
    "timestral-209m-mv-cal-ctx30-f-solar": ("#2a78d6", "dashdot", 2.0),
    "timestral-209m-mv-cal-ctx30-f-tend": ("#9ec5f4", "dot", 2.0),
    "timestral-209m-mv-cal-arome-archived-f-resid": ("#0f7a54", "dot", 2.0),
    "timestral-209m-mv-cal-arome-archived-f-atmos": ("#2e9c78", "dashdot", 2.0),
    "timestral-209m-mv-cal-arome-archived-f-solar": ("#1baf7a", "dashdot", 2.0),
    "timestral-209m-mv-cal-arome-archived-f-tend": ("#7fd1b2", "dashdot", 2.0),
    "arome-biascorr": ("#f29a6b", "dash", 2.2),
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
# Colour family per model family for ids without an explicit STYLE entry (next family: TFC t0, browns).
FAMILY_STYLE = {"t0-": ("#a0522d", "solid", 2.4)}
for _mid, _colour in [("analogue-level", "#718096"), ("analogue-shape", "#4a5568"),
                      ("timestral-209m", "#184f95"), ("t0-gamma", "#805ad5")]:
    STYLE[_mid] = (_colour, "solid", 2.4)
for _base in ("timestral-209m", "t0-gamma"):
    for _method in ("level", "shape"):
        _mid = f"{_base}-rag-{_method}"
        MODELS[_mid] = (f"{MODELS[_base][0]} + RAG · {_method}",
                        "Retrieved pre-2026 analogue trajectories passed as native known covariates; no fine-tuning.")
        STYLE[_mid] = (STYLE[_base][0], "dash" if _method == "shape" else "dot", 2.4)
# Systems whose inputs differ in kind (not just in model): hatched in comparisons, explained in the summary.
DIFFERENT_INPUT = frozenset({"timestral-28m-mv-cal-arome-substobs", "timestral-209m-mv-cal-arome-substobs",
                             "timestral-28m-mv-cal-arome"})


def model_name(mid: str) -> str:
    if mid.startswith(("timestral-dev ", "t0-dev ")):
        _, ck, target, conditioning, _ = mid.split()
        method = "AROME error correction" if target == "arome_error" else \
            "direct + AROME" if conditioning != "none" else "direct"
        return f"{ck.replace('-', ' ')} · {method} (development)"
    if mid.startswith("exploratory "):
        checkpoint = mid.split()[1].replace("Timestral-", "Timestral ")
        method = "AROME error correction" if "+AROME error correction" in mid else \
            "direct + AROME" if " +AROME " in mid else "direct"
        if "+RAG " in mid:
            method = "RAG · " + mid.split("+RAG ", 1)[1].split()[0]
            if "+error-cases" in mid:
                method = "AROME-error " + method
            if "+AROME error correction" in mid:
                method += " · residual prediction"
            elif " +AROME " in mid:
                method += " + AROME"
        return f"{checkpoint} · {method}"
    return MODELS.get(mid, (mid, ""))[0]


def model_description(mid: str) -> str:
    if mid.startswith(("timestral-dev ", "t0-dev ")):
        return "Explicit full-suite development run with recorded configuration; inference only, no local fine-tuning."
    if mid.startswith("exploratory"):
        if "+error-cases" in mid:
            return ("Retrieved verified errors from earlier issued AROME cases, with archived AROME conditioning. "
                    "Only fully verified cases before the query qualify; no fine-tuning or demonstrated improvement claim.")
        if "+RAG " in mid:
            return ("Historical measured analogue trajectories from a verified pre-2026 index, supplied as native "
                    "known covariates. Hourly observations delayed 3 h; no fine-tuning; outside frozen benchmarks.")
        if "+AROME error correction" in mid:
            return ("Issued AROME + Timestral-predicted observation-minus-AROME error; archived error history, "
                    "forecast lead and selected inputs. Inference only, no local fine-tuning; outside the frozen benchmark.")
        return "Run in this session, outside the frozen benchmark; not saved or scored."
    return MODELS.get(mid, ("", ""))[1]


def model_style(mid: str) -> tuple[str, str, float]:
    if mid.startswith(("timestral-dev ", "t0-dev ")):
        _, _, target, conditioning, _ = mid.split()
        if mid.startswith("t0-dev "):
            return ("#805ad5", "solid", 2.8)
        return ("#6250d6" if target == "arome_error" else "#0f7a54" if conditioning != "none" else "#184f95", "solid", 2.8)
    if mid.startswith("exploratory"):
        if mid.startswith("exploratory t0-gamma "):
            return ("#805ad5", "longdash", 2.2)
        if "+AROME error correction" in mid:
            return ("#6250d6", "solid", 2.8)
        return ("#0f7a54" if " +AROME " in mid else "#184f95", "longdash", 2.2)
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
    "rag-arome-error-2026-v1": {
        "name": "AROME-error RAG · verified earlier cases",
        "status": "Development research",
        "short": "Orly and Blagnac · genuine issued AROME · causal error retrieval",
        "scope": "Past weather, recent verified errors and the issued forecast identify similar earlier AROME cases.",
        "caveat": "Only completely verified earlier cases qualify. No fine-tuning or demonstrated improvement claim.",
        "headline": [], "reference": "arome-raw", "default_systems": [], "band": "",
    },
    "rag-weather-2026-v1": {
        "name": "Weather RAG · 2026 replay",
        "status": "Development research",
        "short": "2022–2025 retrieval archive · daily 2026 cases · separate 24/48 h runs",
        "scope": "Hourly station observations; rolling contexts; immutable archive, index and evaluation artifacts.",
        "caveat": "Missing issued forecasts remain unavailable; publication times use explicit replay assumptions.",
        "headline": ["arome-raw", "timestral-209m", "t0-gamma"],
        "reference": "arome-raw",
        "default_systems": ["arome-raw", "timestral-209m", "t0-gamma"],
        "band": "",
    },
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

EXPERIMENTS["hourly-explorer-v1"] = {
    "name": "Hourly exploration · 2022–latest", "status": "Exploratory",
    "scope": "Five stations, hourly UTC observations from January 2022 through each station's last recorded hour.",
    "short": "Hourly observations · 2022–latest · forecasts run only when requested",
    "caveat": "QC gaps remain missing. Historical observations assume availability at validity; no held-out claim.",
    "headline": [], "default_systems": [], "band": None, "reference": None, "compare": {},
}


EXPERIMENTS["hourly-2026-v2"] = {
    "name": "2026 · hourly 48-hour windows · Orly and Blagnac", "status": "Development replay",
    "scope": "Every UTC hour from January 2026 through the last measured record; 1–48 h leads.",
    "caveat": "Overlapping windows; observations and issued AROME use a 3-hour replay availability assumption.",
    "headline": ["persistence"], "reference": "persistence", "default_systems": ["persistence"],
    "band": None, "sample": "Hourly 2026 station observations", "benchmark_only": True,
}
for _new, _old in [("rag-weather-2026-hourly-v2", "rag-weather-2026-v1"),
                   ("rag-arome-error-2026-hourly-v2", "rag-arome-error-2026-v1")]:
    EXPERIMENTS[_new] = {**EXPERIMENTS[_old], "name": EXPERIMENTS[_old]["name"] + " · hourly sliding windows"}

EXPERIMENTS["hourly-2026-v3"] = {**EXPERIMENTS["hourly-2026-v2"],
                                "caveat": "All available three-hourly AROME issues; incomplete windows remain unavailable."}
for _new, _old in [("rag-weather-2026-hourly-v3", "rag-weather-2026-hourly-v2"),
                   ("rag-arome-error-2026-hourly-v3", "rag-arome-error-2026-hourly-v2")]:
    EXPERIMENTS[_new] = {**EXPERIMENTS[_old], "name": EXPERIMENTS[_old]["name"] + " · expanded issues"}
