from __future__ import annotations

import math
from typing import Sequence

import torch


def _as_matrix(sim: torch.Tensor) -> torch.Tensor:
    if sim.dim() != 2:
        raise ValueError(
            f"similarity must be 2D [n_pred, n_ref], got shape {tuple(sim.shape)}"
        )
    return sim


def alignment_max_sims(sim: torch.Tensor) -> torch.Tensor:
    """Per-ref max similarity over preds. ``sim`` [n_pred, n_ref] → [n_ref]."""
    sim = _as_matrix(sim)
    if sim.numel() == 0:
        return torch.empty(0)
    return sim.max(dim=0).values


def hallucination_max_sims(sim: torch.Tensor) -> torch.Tensor:
    """Per-pred max similarity over refs. ``sim`` [n_pred, n_ref] → [n_pred]."""
    sim = _as_matrix(sim)
    if sim.numel() == 0:
        return torch.empty(0)
    return sim.max(dim=1).values


def alignment_recall(sim: torch.Tensor, threshold: float) -> float:
    """ALIGN: fraction of refs with max similarity >= threshold."""
    best = alignment_max_sims(sim)
    if best.numel() == 0:
        return 0.0
    return (best >= threshold).float().mean().item()


def hallucination_rate(sim: torch.Tensor, threshold: float) -> float:
    """HAL: fraction of preds with max similarity < threshold."""
    best = hallucination_max_sims(sim)
    if best.numel() == 0:
        return 0.0
    return (best < threshold).float().mean().item()


def exaggeration_positive_deltas(
    gen_scores: torch.Tensor, reference_scores: torch.Tensor
) -> torch.Tensor:
    """Per-candidate ``max(gen - reference, 0)``."""
    gen_scores = gen_scores.flatten().float()
    reference_scores = reference_scores.flatten().float()
    return torch.clamp(gen_scores - reference_scores, min=0.0)


def exaggeration_delta(
    gen_scores: torch.Tensor, reference_scores: torch.Tensor
) -> float:
    """EXAG: mean of positive (gen - reference) gaps over candidates."""
    positive = exaggeration_positive_deltas(gen_scores, reference_scores)
    if positive.numel() == 0:
        return 0.0
    return positive.mean().item()


def descriptor_diversity(sims: Sequence[torch.Tensor], threshold: float) -> float:
    """DDIV: normalised entropy of per-ref coverage counts across images."""
    if len(sims) < 2:
        raise ValueError("DDIV requires >= 2 images.")
    n_ref = sims[0].shape[1]
    if n_ref == 0:
        return 0.0
    counts = torch.zeros(n_ref)
    for sim in sims:
        covered = alignment_max_sims(sim) >= threshold
        counts = counts + covered.float()
    total = counts.sum().item()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    entropy = -(p * p.log()).sum().item()
    return float(entropy / math.log(n_ref)) if n_ref > 1 else 0.0


def semantic_diversity(pooled_align: float, mean_single_align: float) -> float:
    """SDIV: pooled ALIGN minus mean single-image ALIGN."""
    return float(pooled_align - mean_single_align)
