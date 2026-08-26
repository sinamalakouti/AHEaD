"""Embedding cosine-similarity matcher (default for ALIGN/HAL/DDIV/SDIV)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from ahead.matchers.base import Matcher


class EmbeddingMatcher(Matcher):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache: bool = True,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, torch.Tensor] = {} if cache else None
        self._model = None  # lazy

    def _lazy_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        texts = list(texts)
        if not texts:
            return torch.empty(0, 0, device=self.device)

        if self._cache is not None:
            missing = [t for t in texts if t not in self._cache]
            if missing:
                emb = self._lazy_model().encode(
                    missing, convert_to_tensor=True, normalize_embeddings=True
                )
                for t, e in zip(missing, emb):
                    self._cache[t] = e.to(self.device)
            return torch.stack([self._cache[t] for t in texts])

        emb = self._lazy_model().encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        )
        return emb.to(self.device)

    def similarity(
        self,
        pred: Sequence[str],
        ref: Sequence[str],
        image: Optional[str] = None,  # unused for text embeddings
    ) -> torch.Tensor:
        pred_emb = self.embed(pred)
        ref_emb = self.embed(ref)
        if pred_emb.numel() == 0 or ref_emb.numel() == 0:
            return torch.empty(len(pred), len(ref), device=self.device)
        return pred_emb @ ref_emb.T  # cosine, since both are normalised
