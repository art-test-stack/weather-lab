"""Isolated experimental transform; the registered t0 weather adapter stays unchanged."""
from dataclasses import replace

import numpy as np
import pandas as pd

from weather_lab.forecast import QUANTILE_LEVELS
from weather_lab.t0_adapter import T0


class ResidualT0(T0):
    kind = "experimental_t0_arome_error"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("arome", "missing")
        if kwargs["arome"] != "missing" or kwargs.get("calendar", False):
            raise ValueError("This experiment requires forecast-only AROME and calendar off")
        super().__init__(*args, **kwargs)
        self.params.update(prediction_target="arome_error", experimental_wrapper_version=1,
                           residual_history="latest qualifying archived forecast", forecast_lead_input=True)

    def residual_inputs(self, batch):
        if batch.arome is None or batch.arome_history is None:
            raise ValueError("Residual wrapper needs issued and archived AROME inputs")
        length = self.context_steps(batch.step_hours)
        idx = [batch.variables.index(v) for v in self.variables]
        baseline = batch.arome[:, idx]
        if not np.isfinite(baseline).all():
            raise ValueError("Residual wrapper needs every issued forecast anchor")
        errors = batch.history[:, idx, -length:] - batch.arome_history[:, idx, -length:]
        if not np.isfinite(errors).any(axis=-1).all():
            raise ValueError("Residual wrapper refuses an entirely missing error history")
        runs = pd.to_datetime(batch.cases["arome_run"], utc=True).to_numpy(dtype="datetime64[ns]")
        if np.isnat(runs).any() or (runs > batch.decision_origins).any():
            raise ValueError("Residual wrapper source run is unavailable at the decision time")
        if "arome_available_at" in batch.cases:
            available = pd.to_datetime(batch.cases.arome_available_at, utc=True).to_numpy(dtype="datetime64[ns]")
            if np.isnat(available).any() or (available > batch.decision_origins).any():
                raise ValueError("Residual wrapper source run is unavailable at the decision time")
        past = batch.arome_history_lead_h[:, -length:]
        has = np.isfinite(errors).any(axis=1)
        history_runs = batch.arome_history_run[:, -length:]
        if (np.isnat(history_runs[has]).any() or not np.isfinite(past[has]).all()
                or (past[has] <= 0).any() or (history_runs[has] >= batch.history_times(length)[has]).any()):
            raise ValueError("Residual history lacks causal forecast provenance")
        future = (batch.future_times - runs[:, None]) / np.timedelta64(1, "h")
        if not np.isfinite(future).all() or (future <= 0).any():
            raise ValueError("Residual wrapper needs positive actual issued-run leads")
        leads = np.concatenate([np.where(has, past, np.nan), future], axis=1)[:, None]
        return errors.astype(np.float32), baseline, leads.astype(np.float32)

    def build_inputs(self, batch):
        errors, _, leads = self.residual_inputs(batch)
        context, known, report = super().build_inputs(batch)
        context[:, :len(self.variables)] = errors
        known = np.concatenate([known, leads], axis=1)
        return context, known, {**report, "known_rows": known.shape[1],
                                "target": "observation minus latest qualifying archived AROME",
                                "residual_observed_steps": np.isfinite(errors).sum(axis=-1).tolist(),
                                "inverse_transform": "add selected issued AROME to every weather-target quantile"}

    def predict(self, batch):
        residual = super().predict(batch)
        _, baseline, _ = self.residual_inputs(batch)
        quantiles = residual.quantiles + baseline[..., None]
        return replace(residual, point=quantiles[..., QUANTILE_LEVELS.index(.5)], quantiles=quantiles,
                       info={**residual.info, "prediction_target": "arome_error", "experimental_wrapper_version": 1})
