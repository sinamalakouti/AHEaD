"""Sentence-embedding backbone."""

from __future__ import annotations

from ahead.matchers.embedding import EmbeddingMatcher


def get_embedder(name: str = "sentence-transformers/all-MiniLM-L6-v2", device=None, **kwargs):
    return EmbeddingMatcher(model_name=name, device=device, **kwargs)
