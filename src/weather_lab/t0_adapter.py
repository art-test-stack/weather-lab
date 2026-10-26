"""Native tfc-t0 adapter; historical covariates are additional joint target variates."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from weather_lab import features as features_module
from weather_lab.forecast import Adapter, CaseBatch, ForecastBatch, QUANTILE_LEVELS, calendar_features


class T0(Adapter):
    kind = family = "t0"
    output = "quantiles"
    ADAPTER_VERSION = 1
    PREPROCESSING_VERSION = 1

    def __init__(self, id: str, variables: list[str], targets: list[str], checkpoint: str, revision: str,
                 context_days: int = 30, calendar: bool = False, arome: str | None = None,
                 arome_fields: list[str] | None = None, batch_size: int = 8, device: str = "cpu",
                 features: list[str] | dict | None = None, trim_leading_missing: bool = False,
                 symmetric_averaging: bool = False, clip_nonnegative: bool = False,
                 snap_integers: bool = False, prediction_target: str = "weather", **params):
        if clip_nonnegative or snap_integers:
            raise ValueError("t0 weather targets in Celsius/hPa/signed wind require clipping and integer snapping off")
        if prediction_target != "weather":
            raise ValueError("t0 adapter supports raw weather targets; fixed-lead residual targets need their own builder")
        if arome not in (None, "archived", "missing"):
            raise ValueError("t0 AROME history must be archived or explicitly missing; observation substitution is unsupported")
        if not targets or not set(targets) <= set(variables) or context_days < 1 or batch_size < 1:
            raise ValueError("t0 needs valid target variables, a positive context and batch size")
        self.feature_spec = features_module.normalize(features or [])
        self.features = list(self.feature_spec)
        self.checkpoint, self.revision, self.device = checkpoint, revision, device
        self.context_days, self.calendar, self.arome_mode = context_days, calendar, arome
        self.arome_fields = list(arome_fields or [])
        if not set(self.arome_fields) <= set(variables):
            raise ValueError("t0 AROME fields must belong to the batch variables")
        self.batch_size = batch_size
        self.options = {"trim_leading_missing": bool(trim_leading_missing),
                        "symmetric_averaging": bool(symmetric_averaging),
                        "clip_nonnegative": False, "snap_integers": False}
        super().__init__(id, targets, checkpoint=checkpoint, context_days=context_days, calendar=calendar,
                         arome=arome, arome_fields=self.arome_fields, features=self.features,
                         feature_definitions=features_module.definition(self.feature_spec),
                         inference_options=self.options, **params)
        needs = {dep for name in self.features for dep in features_module.GROUPS[name].needs}
        self.inputs = frozenset({"history"} | ({"calendar"} if calendar else set())
                                | ({"arome"} if arome or "arome_history" in needs else set())
                                | ({"features"} if self.features else set()))

    def context_steps(self, step_hours: int) -> int:
        if self.context_days * 24 % step_hours:
            raise ValueError("t0 context must align with the observation cadence")
        return self.context_days * 24 // step_hours

    def build_inputs(self, batch: CaseBatch) -> tuple[np.ndarray, np.ndarray | None, dict]:
        length = self.context_steps(batch.step_hours)
        if batch.history.shape[-1] < length:
            raise ValueError("t0 observation context is shorter than requested")
        index = [batch.variables.index(v) for v in self.variables]
        context = batch.history[:, index, -length:].copy()
        if not np.isfinite(context).any(axis=-1).all():
            raise ValueError("t0 refuses an entirely missing weather target row")
        known = []
        if self.calendar:
            known.append(calendar_features(np.concatenate([batch.history_times(length), batch.future_times], axis=1)))
        if self.arome_mode:
            if batch.arome is None or not self.arome_fields:
                raise ValueError("t0 conditioning needs issued AROME and explicit fields")
            for variable in self.arome_fields:
                j = batch.variables.index(variable)
                future = batch.arome[:, j:j + 1]
                if not np.isfinite(future).all():
                    raise ValueError(f"t0 issued AROME {variable} is missing; native NaN-to-zero is not an archive fallback")
                if self.arome_mode == "archived":
                    if batch.arome_history is None:
                        raise ValueError("t0 archived conditioning needs vintage-checked AROME history")
                    past = batch.arome_history[:, j:j + 1, -length:]
                else:
                    past = np.full((len(context), 1, length), np.nan, np.float32)
                known.append(np.concatenate([past, future], axis=2))
        feature_inputs = features_module.build_inputs(self.feature_spec, batch, length)
        if feature_inputs.past is not None:
            if not np.isfinite(feature_inputs.past).any(axis=-1).all():
                raise ValueError("t0 selected historical feature row is entirely missing")
            context = np.concatenate([context, feature_inputs.past], axis=1)
        if feature_inputs.known is not None:
            if not np.isfinite(feature_inputs.known[:, :, -batch.horizon:]).all():
                raise ValueError("t0 known-future feature horizon is incomplete; missing inputs cannot become zero")
            known.append(feature_inputs.known)
        covariates = np.concatenate(known, axis=1).astype(np.float32) if known else None
        report = {**feature_inputs.report, "native_past_semantics": "extra jointly forecast variates; outputs discarded",
                  "native_known_semantics": "future_covariates over context+horizon; native masks retained",
                  "target_rows": len(self.variables), "joint_context_rows": context.shape[1],
                  "known_rows": 0 if covariates is None else covariates.shape[1]}
        return context.astype(np.float32), covariates, report

    def input_identity(self, batch: CaseBatch) -> str:
        context, known, _ = self.build_inputs(batch)
        digest = hashlib.sha256(json.dumps({"adapter_version": self.ADAPTER_VERSION,
                                            "runtime": "tfc-t0==0.6.0", "revision": self.revision,
                                            "device": self.device, "precision": "float32", "params": self.params,
                                            "variables": self.variables, "quantiles": QUANTILE_LEVELS},
                                           sort_keys=True).encode())
        for value in (context, known, batch.origins, batch.future_times):
            if value is None:
                digest.update(b"none")
            else:
                digest.update(str(value.shape).encode())
                digest.update(str(value.dtype).encode())
                digest.update(np.ascontiguousarray(value).tobytes())
        digest.update("|".join(batch.cases.station).encode())
        return digest.hexdigest()

    def predict(self, batch: CaseBatch) -> ForecastBatch:
        import torch
        from weather_lab.models import MODELS, resolve

        context, known, report = self.build_inputs(batch)
        item = MODELS.get(resolve(self.checkpoint, self.revision), self.device)
        levels = tuple(round(float(q), 6) for q in item.forecaster.config.quantile_levels)
        if levels != QUANTILE_LEVELS:
            raise ValueError("t0 checkpoint native quantile grid differs from the benchmark; mapping is not implemented")
        out = []
        started = time.perf_counter()
        with item.lock, torch.inference_mode():
            for start in range(0, len(context), self.batch_size):
                sl = slice(start, start + self.batch_size)
                result = item.forecaster.predict(context[sl], horizon=batch.horizon, quantile_levels=QUANTILE_LEVELS,
                                                 future_covariates=None if known is None else known[sl],
                                                 context_length=context.shape[-1], **self.options)
                if tuple(round(float(q), 6) for q in result.quantile_levels) != QUANTILE_LEVELS:
                    raise ValueError("t0 returned an unexpected quantile grid")
                q = result.quantiles.detach().cpu().numpy()
                shape = (len(context[sl]), context.shape[1], batch.horizon, len(QUANTILE_LEVELS))
                if q.shape == (shape[0] * shape[1], *shape[2:]):
                    q = q.reshape(shape)
                if q.shape != shape:
                    raise ValueError(f"t0 native output shape {q.shape} differs from {shape}")
                out.append(q[:, :len(self.variables)])
        quantiles = np.concatenate(out)
        return ForecastBatch(self.id, f"{self.checkpoint}@{self.revision}", "quantiles", self.variables,
                             quantiles[..., QUANTILE_LEVELS.index(.5)], quantiles, QUANTILE_LEVELS,
                             info={**item.info, "adapter_version": self.ADAPTER_VERSION,
                                   "native_inputs": report, "inference_options": self.options,
                                   "input_identity": self.input_identity(batch),
                                   "predict_s": time.perf_counter() - started, "n_cases": len(context),
                                   "context_steps": context.shape[-1], "batch_size": self.batch_size})
