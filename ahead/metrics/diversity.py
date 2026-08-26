"""DDIV (Eq. 6) and SDIV (Eq. 7) — multi-image only."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import kernels
from ahead.matchers.base import Matcher
from ahead.metrics.common import (
    MULTI,
    ExtractFn,
    ImagePath,
    as_mode,
    resolve_preds,
    resolve_refs_multi,
)
from ahead.types import DescriptorSet, MetricResult, normalise_descriptors


def _reference_dims(ref: dict) -> List[str]:
    return [d for d in ref.keys() if len(ref[d]) > 0]


def compute_descriptor_diversity(
    preds_per_image: List[dict],
    reference_descriptors: dict,
    matcher: Matcher,
    threshold: float,
) -> MetricResult:
    ref = normalise_descriptors(reference_descriptors)
    preds_per_image = [normalise_descriptors(p) for p in preds_per_image]

    if len(preds_per_image) < 2:
        raise ValueError("DDIV requires >= 2 images.")

    per_dim: Dict[str, float] = {}
    for dim in _reference_dims(ref):
        ref_dim = list(dict.fromkeys(ref[dim]))
        sims = [matcher.similarity(img.get(dim, []), ref_dim) for img in preds_per_image]
        per_dim[dim] = kernels.descriptor_diversity(sims, threshold)

    value = float(np.mean(list(per_dim.values()))) if per_dim else 0.0
    return MetricResult(name="ddiv", value=value, per_dimension=per_dim)


def compute_semantic_diversity(
    preds_per_image: List[dict],
    reference_descriptors: dict,
    matcher: Matcher,
    threshold: float,
) -> MetricResult:
    ref = normalise_descriptors(reference_descriptors)
    preds_per_image = [normalise_descriptors(p) for p in preds_per_image]

    if len(preds_per_image) < 2:
        raise ValueError("SDIV requires >= 2 images.")

    per_dim: Dict[str, float] = {}
    for dim in _reference_dims(ref):
        ref_dim = list(dict.fromkeys(ref[dim]))
        pooled_preds = [d for img in preds_per_image for d in img.get(dim, [])]
        pooled_recall = kernels.alignment_recall(
            matcher.similarity(pooled_preds, ref_dim), threshold
        )
        single_recalls = [
            kernels.alignment_recall(matcher.similarity(img.get(dim, []), ref_dim), threshold)
            for img in preds_per_image
        ]
        mean_single = float(np.mean(single_recalls)) if single_recalls else 0.0
        per_dim[dim] = kernels.semantic_diversity(pooled_recall, mean_single)

    value = float(np.mean(list(per_dim.values()))) if per_dim else 0.0
    return MetricResult(name="sdiv", value=value, per_dimension=per_dim)


class Ddiv:
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
        mode: str = MULTI,
    ) -> MetricResult:
        if as_mode(mode, default=MULTI) != MULTI:
            raise ValueError("DDIV only supports mode='multi-image'")
        pred_list = resolve_preds(preds, images, concept, self.extract)
        if len(pred_list) < 2:
            raise ValueError("DDIV needs at least 2 images/preds.")
        return compute_descriptor_diversity(
            pred_list, resolve_refs_multi(refs), self.matcher, self.threshold
        )


class Sdiv:
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
        mode: str = MULTI,
    ) -> MetricResult:
        if as_mode(mode, default=MULTI) != MULTI:
            raise ValueError("SDIV only supports mode='multi-image'")
        pred_list = resolve_preds(preds, images, concept, self.extract)
        if len(pred_list) < 2:
            raise ValueError("SDIV needs at least 2 images/preds.")
        return compute_semantic_diversity(
            pred_list, resolve_refs_multi(refs), self.matcher, self.threshold
        )
