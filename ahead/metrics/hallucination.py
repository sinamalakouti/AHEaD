"""HAL (Eq. 2): fraction of predicted descriptors with no reference match."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import kernels
from ahead.matchers.base import Matcher
from ahead.metrics.common import (
    SINGLE,
    ExtractFn,
    ImagePath,
    as_mode,
    resolve_preds,
    resolve_refs_multi,
    resolve_refs_single,
)
from ahead.types import DescriptorSet, MetricResult, normalise_descriptors


def compute_hallucination(
    preds_per_image: List[dict],
    reference_descriptors: dict,
    matcher: Matcher,
    threshold: float,
) -> MetricResult:
    ref = normalise_descriptors(reference_descriptors)
    preds_per_image = [normalise_descriptors(p) for p in preds_per_image]

    per_dim: Dict[str, float] = {}
    per_desc: Dict[str, float] = {}
    per_desc_by_dim: Dict[str, Dict[str, float]] = {}

    for dim in [d for d in ref.keys() if len(ref[d]) > 0]:
        ref_dim = list(dict.fromkeys(ref[dim]))
        pooled_preds = [d for img in preds_per_image for d in img.get(dim, [])]
        unique_preds = list(dict.fromkeys(pooled_preds))
        sim = matcher.similarity(unique_preds, ref_dim)
        soft = kernels.hallucination_max_sims(sim)
        per_dim[dim] = (
            (soft < threshold).float().mean().item() if soft.numel() else 0.0
        )
        dim_scores = {t: float(soft[i].item()) for i, t in enumerate(unique_preds)}
        per_desc_by_dim[dim] = dim_scores
        for t, s in dim_scores.items():
            if t not in per_desc or s > per_desc[t]:
                per_desc[t] = s

    value = float(np.mean(list(per_dim.values()))) if per_dim else 0.0
    return MetricResult(
        name="hal",
        value=value,
        per_dimension=per_dim,
        per_descriptor=per_desc,
        per_descriptor_by_dimension=per_desc_by_dim,
    )


def _as_score_dict(scores: Union[MetricResult, Dict[str, float]]) -> Dict[str, float]:
    if isinstance(scores, MetricResult):
        return dict(scores.per_descriptor)
    return dict(scores)


class Hal:
    def __init__(
        self,
        matcher: Matcher,
        threshold: float = 0.52,
        extract: Optional[ExtractFn] = None,
    ):
        self.matcher = matcher
        self.threshold = threshold
        self.extract = extract

    def score(
        self,
        refs,
        preds: Optional[Sequence[DescriptorSet]] = None,
        images: Optional[Sequence[ImagePath]] = None,
        concept: Optional[str] = None,
        mode: str = SINGLE,
    ) -> Union[MetricResult, List[MetricResult]]:
        mode = as_mode(mode)
        pred_list = resolve_preds(preds, images, concept, self.extract)
        if mode == SINGLE:
            ref_list = resolve_refs_single(refs, len(pred_list))
            return [
                compute_hallucination([p], r, self.matcher, self.threshold)
                for p, r in zip(pred_list, ref_list)
            ]
        return compute_hallucination(
            pred_list, resolve_refs_multi(refs), self.matcher, self.threshold
        )

    def rank(
        self,
        scores: Union[MetricResult, Dict[str, float], Sequence[MetricResult]],
        k: Optional[int] = None,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Descriptors sorted by higher HAL. Soft scores are max-sim; rank uses ``1 - soft``."""
        if isinstance(scores, Sequence) and not isinstance(scores, (dict, MetricResult)):
            return [self.rank(s, k=k) for s in scores]
        soft = _as_score_dict(scores)
        ranked = sorted(
            ((t, 1.0 - s) for t, s in soft.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked if k is None else ranked[:k]
