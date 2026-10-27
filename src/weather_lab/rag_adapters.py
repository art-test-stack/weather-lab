"""Retrieval-only references and retrieved trajectories as native known covariates."""
from __future__ import annotations

import numpy as np

from weather_lab.forecast import Adapter, ForecastBatch, Timestral
from weather_lab.t0_adapter import T0


class Analog(Adapter):
    kind = "analogue"
    revision = "window-retrieval-v1"

    def __init__(self, ident, index, method="shape", k=5):
        self.index, self.method, self.k = index, method, k
        super().__init__(ident, index.meta["variables"], index_id=index.ident, method=method, k=k)

    def predict(self, batch):
        rows, audit = self.index.retrieve(batch, self.method, self.k)
        return ForecastBatch(self.id, self.revision, "point", self.variables, rows[:, :, -batch.horizon:],
                             info={"retrieval": audit})


class Retrieved:
    def __init__(self, *args, index, method="shape", k=5, **kwargs):
        self.index, self.method, self.k = index, method, k
        self.audit = []
        super().__init__(*args, **kwargs)
        self.params.update(index_id=index.ident, retrieval_method=method, retrieval_k=k,
                           retrieval_role="historical analogue trajectory as known covariates", rag_version=1)

    def build_inputs(self, batch):
        native = super().build_inputs(batch)
        rows, self.audit = self.index.retrieve(batch, self.method, self.k)
        if self.variables != self.index.meta["variables"]:
            raise ValueError("RAG target order must equal the indexed variable order")
        if rows.shape[-1] != native[0].shape[-1] + batch.horizon:
            raise ValueError("retrieved and model contexts differ")
        known = rows if native[1] is None else np.concatenate([native[1], rows], axis=1)
        return (native[0], known, *native[2:])

    def predict(self, batch):
        result = super().predict(batch)
        result.info.update(retrieval=self.audit, rag_version=1, index_id=self.index.ident)
        return result


class RetrievedTimestral(Retrieved, Timestral):
    pass


class RetrievedT0(Retrieved, T0):
    pass


def foundation(name, index, method=None, k=5, device="cpu", calendar=True):
    from weather_lab.models import CHECKPOINTS

    ck = CHECKPOINTS[name]
    classes = {("timestral", False): Timestral, ("timestral", True): RetrievedTimestral,
               ("t0", False): T0, ("t0", True): RetrievedT0}
    kind = classes[(ck.family, method is not None)]
    ident = f"{name.lower()}-rag-{method}" if method else name.lower()
    kwargs = {"index": index, "method": method, "k": k} if method else {}
    return kind(ident, variables=index.meta["variables"], targets=index.meta["variables"], checkpoint=ck.repo,
                revision=ck.revision, context_days=index.meta["context_hours"] // 24,
                calendar=calendar, device=device, batch_size=8, **kwargs)
