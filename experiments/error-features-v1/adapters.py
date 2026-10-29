"""Experiment-local, model-independent information rows with native translations."""
import importlib.util

import numpy as np
import pandas as pd

from weather_lab.common import ROOT
from weather_lab.forecast import Timestral

spec = importlib.util.spec_from_file_location("prior_t0_error_wrapper",
    ROOT / "experiments/t0-error-wrapper-v1/error_wrapper.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ResidualT0 = module.ResidualT0
FIELDS = ("relative_humidity_2m", "cloud_cover", "shortwave_radiation")


class ExtraInputs:
    def __init__(self, wind, issued):
        self.wind, self.issued = wind, issued

    def rows(self, batch, length):
        past, known = None, None
        if self.wind:
            if not {"u", "v"} <= set(batch.variables):
                raise ValueError("Past wind requires original observed u/v components")
            past = batch.history[:, [batch.variables.index(v) for v in ("u", "v")], -length:].copy()
            if not np.isfinite(past).any(axis=-1).all():
                raise ValueError("Past wind row is entirely missing")
        if self.issued is not None:
            future = np.full((len(batch.cases), 3, batch.horizon), np.nan, np.float32)
            for i, case in enumerate(batch.cases.itertuples()):
                run = pd.Timestamp(case.arome_run)
                origin = pd.Timestamp(batch.decision_origins[i], tz="UTC")
                source = self.issued[self.issued.station.eq(case.station) & self.issued.run.eq(run)]
                if source.empty or source.available_at.isna().any() or source.available_at.gt(origin).any():
                    raise ValueError("Issued weather absent or unavailable at the decision")
                if not source["product"].eq("meteofrance_arome_france").all():
                    raise ValueError("Issued weather uses a different product")
                if source.duplicated(["variable", "valid_time"]).any():
                    raise ValueError("Duplicate issued weather values")
                times = pd.to_datetime(batch.future_times[i], utc=True)
                for j, field in enumerate(FIELDS):
                    future[i, j] = source[source.variable.eq(field)].set_index("valid_time").value.reindex(times)
            if not np.isfinite(future).all():
                raise ValueError("Issued weather lacks a requested future value")
            known = np.concatenate([np.full((*future.shape[:2], length), np.nan, np.float32), future], axis=-1)
        return past, known


class ExtraTimestral(Timestral):
    def __init__(self, *args, extras, extra_identity, **kwargs):
        super().__init__(*args, **kwargs)
        self.extras = extras
        self.params["extra_information_v1"] = extra_identity

    def feature_rows(self, batch):
        past, known, report = super().feature_rows(batch)
        wind, weather = self.extras.rows(batch, self.context_steps(batch.step_hours))
        if wind is not None:
            past = wind if past is None else np.concatenate([past, wind], axis=1)
        if weather is not None:
            known = weather if known is None else np.concatenate([known, weather], axis=1)
        return past, known, {**report, "extra_information_v1": self.params["extra_information_v1"]}


class ExtraT0(ResidualT0):
    def __init__(self, *args, extras, extra_identity, **kwargs):
        super().__init__(*args, **kwargs)
        self.extras = extras
        self.params["extra_information_v1"] = extra_identity

    def build_inputs(self, batch):
        context, known, report = super().build_inputs(batch)
        wind, weather = self.extras.rows(batch, self.context_steps(batch.step_hours))
        if wind is not None:
            context = np.concatenate([context, wind], axis=1)
        if weather is not None:
            known = np.concatenate([known, weather], axis=1)
        return context, known, {**report, "joint_context_rows": context.shape[1], "known_rows": known.shape[1],
                                "extra_information_v1": self.params["extra_information_v1"]}
