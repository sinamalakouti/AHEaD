"""AHEaD metrics: ALIGN, HAL, EXAG, DDIV, SDIV."""

from __future__ import annotations

from typing import List, Optional, Sequence

from ahead.matchers.base import ImageTextScorer, Matcher, resolve_matcher
from ahead.metrics.alignment import Align, compute_alignment
from ahead.metrics.diversity import (
    Ddiv,
    Sdiv,
    compute_descriptor_diversity,
    compute_semantic_diversity,
)
from ahead.metrics.exaggeration import Exag, compute_exaggeration
from ahead.metrics.hallucination import Hal, compute_hallucination
from ahead.metrics.stereotypes import StereotypeCandidateGenerator


class AheadMetrics:
    def __init__(
        self,
        matcher: "Matcher | str | None" = "embedding",
        threshold: float = 0.52,
        scorer: Optional[ImageTextScorer] = None,
        extractor: "object | str | None" = None,
        **matcher_kwargs,
    ):
        self.matcher = resolve_matcher(matcher, **matcher_kwargs)
        self.threshold = threshold
        self.scorer = scorer
        self._extractor = extractor

        self.align = Align(self.matcher, self.threshold, extract=self.extract)
        self.hal = Hal(self.matcher, self.threshold, extract=self.extract)
        self.exag = Exag(self.scorer)
        self.ddiv = Ddiv(self.matcher, self.threshold, extract=self.extract)
        self.sdiv = Sdiv(self.matcher, self.threshold, extract=self.extract)

    def _ensure_extractor(self):
        from ahead.metrics.extraction import DescriptorExtractor

        if self._extractor is None:
            self._extractor = DescriptorExtractor()
        elif isinstance(self._extractor, str):
            self._extractor = DescriptorExtractor(mllm=self._extractor)
        return self._extractor

    def extract(self, images: Sequence[str], concept: str) -> List[dict]:
        return self._ensure_extractor().extract_batch(list(images), concept)


__all__ = [
    "AheadMetrics",
    "Align",
    "Hal",
    "Exag",
    "Ddiv",
    "Sdiv",
    "compute_alignment",
    "compute_hallucination",
    "compute_exaggeration",
    "compute_descriptor_diversity",
    "compute_semantic_diversity",
    "StereotypeCandidateGenerator",
]
