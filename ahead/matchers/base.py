"""Matcher and ImageTextScorer protocols."""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, runtime_checkable

import torch


@runtime_checkable
class Matcher(Protocol):
    def similarity(
        self,
        pred: Sequence[str],
        ref: Sequence[str],
        image: Optional[str] = None,
    ) -> torch.Tensor:
        ...


@runtime_checkable
class ImageTextScorer(Protocol):
    def score(self, images: Sequence[str], texts: Sequence[str]) -> torch.Tensor:
        ...


def resolve_matcher(matcher, **kwargs) -> Matcher:
    if isinstance(matcher, Matcher):
        return matcher
    if matcher in (None, "embedding"):
        from ahead.matchers.embedding import EmbeddingMatcher

        return EmbeddingMatcher(**kwargs)
    raise ValueError(f"unknown matcher {matcher!r}")
