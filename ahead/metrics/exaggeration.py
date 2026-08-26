"""EXAG (Eq. 4): positive over-expression vs the averaged real images."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from . import kernels
from ahead.matchers.base import ImageTextScorer
from ahead.metrics.common import SINGLE, ImagePath, as_mode
from ahead.paths import collect_image_paths
from ahead.types import MetricResult


def _baseline(
    scorer: ImageTextScorer,
    reference_images: Optional[Sequence[str]],
    reference_scores: Optional[torch.Tensor],
    candidates: Sequence[str],
) -> torch.Tensor:
    if reference_scores is not None:
        return torch.as_tensor(reference_scores, dtype=torch.float32).flatten()
    if not reference_images:
        raise ValueError("EXAG needs either reference_images or a reference_scores vector.")
    ref = scorer.score(list(reference_images), list(candidates))
    return torch.as_tensor(ref, dtype=torch.float32).mean(dim=0)


def compute_exaggeration(
    images: Sequence[str],
    candidates: Sequence[str],
    scorer: ImageTextScorer,
    reference_images: Optional[Sequence[str]] = None,
    reference_scores: Optional[torch.Tensor] = None,
) -> MetricResult:
    baseline = _baseline(scorer, reference_images, reference_scores, candidates)
    candidates = list(candidates)

    images = list(images)
    gen = torch.as_tensor(scorer.score(images, candidates), dtype=torch.float32)
    if gen.dim() == 1:
        gen = gen.unsqueeze(0)

    stacked = torch.stack(
        [kernels.exaggeration_positive_deltas(gen[i], baseline) for i in range(gen.shape[0])]
    )
    deltas = stacked.mean(dim=0)
    per_desc = {c: float(deltas[i].item()) for i, c in enumerate(candidates)}
    value = float(deltas.mean().item()) if deltas.numel() else 0.0
    return MetricResult(name="exag", value=value, per_descriptor=per_desc)


def _as_score_dict(scores: Union[MetricResult, Dict[str, float]]) -> Dict[str, float]:
    if isinstance(scores, MetricResult):
        return dict(scores.per_descriptor)
    return dict(scores)


class Exag:
    def __init__(self, scorer: Optional[ImageTextScorer] = None):
        self.scorer = scorer

    def score(
        self,
        images: Sequence[ImagePath],
        candidates: Sequence[str],
        reference_images: Optional[Sequence[ImagePath]] = None,
        reference_scores=None,
        mode: str = SINGLE,
    ) -> Union[MetricResult, List[MetricResult]]:
        if self.scorer is None:
            raise ValueError("EXAG requires scorer=... (e.g. VQAScore) at AheadMetrics init.")
        mode = as_mode(mode)
        if isinstance(images, (str, Path)):
            raise ValueError("images must be a list of file paths")
        gen_paths = [str(p) for p in images]
        if not gen_paths:
            raise ValueError("images= is empty")
        ref_paths = None
        if reference_images is not None:
            if isinstance(reference_images, (str, Path)):
                ref_paths = collect_image_paths(reference_images) or None
            else:
                ref_paths = [str(p) for p in reference_images]
        if mode == SINGLE:
            return [
                compute_exaggeration(
                    [p],
                    candidates,
                    self.scorer,
                    reference_images=ref_paths,
                    reference_scores=reference_scores,
                )
                for p in gen_paths
            ]
        return compute_exaggeration(
            gen_paths,
            candidates,
            self.scorer,
            reference_images=ref_paths,
            reference_scores=reference_scores,
        )

    def rank(
        self,
        scores: Union[MetricResult, Dict[str, float], Sequence[MetricResult]],
        k: Optional[int] = None,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Stereotypes sorted by higher EXAG (higher positive delta). ``k`` keeps the top-k."""
        if isinstance(scores, Sequence) and not isinstance(scores, (dict, MetricResult)):
            return [self.rank(s, k=k) for s in scores]
        items = sorted(
            _as_score_dict(scores).items(), key=lambda x: x[1], reverse=True
        )
        return items if k is None else items[:k]
