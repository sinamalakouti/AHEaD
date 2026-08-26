"""Image–text scorers for EXAG (default: VQAScore)."""

from __future__ import annotations

from typing import Sequence

import torch

from ahead.registry import scorers


@scorers.register("vqascore")
class VQAScoreScorer:
    def __init__(self, model: str = "clip-flant5-xxl", device: str = "cuda"):
        self.device = device
        self.model_name = model
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            import t2v_metrics

            self._model = t2v_metrics.VQAScore(model=self.model_name).to(self.device)
        return self._model

    def score(self, images: Sequence[str], texts: Sequence[str]) -> torch.Tensor:
        model = self._lazy_load()
        return model([str(i) for i in images], list(texts))


def get_scorer(name: str = "vqascore", device: str = "cuda"):
    return scorers.create(name, device=device)


get_ita = get_scorer
