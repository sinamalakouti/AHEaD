"""MLLM-as-a-judge: single-image 1–5 Likert scores."""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence

from ahead.backbones.mllm import BaseMLLM, get_mllm
from ahead.metrics.mllm_judge import prompts

_SCORE_RE = re.compile(r"score is\s*(\d+(?:\.\d+)?)")


def parse_score(response: str) -> Optional[float]:
    m = _SCORE_RE.search((response or "").lower())
    if not m:
        return None
    return float(m.group(1))


class MLLMJudge:
    def __init__(
        self,
        mllm: "BaseMLLM | str" = "internvl",
        device: str = "cuda",
        max_tokens: int = 50,
        temperature: float = 0.0,
        n_samples: int = 2,
        clamp: bool = True,
    ):
        self.mllm = mllm if isinstance(mllm, BaseMLLM) else get_mllm(mllm, device=device)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n_samples = n_samples
        self.clamp = clamp

    def _resolve_metric(self, metric: str) -> str:
        key = metric.lower()
        if key not in prompts.TEMPLATES:
            raise ValueError(
                f"unknown metric {metric!r}; choose from {list(prompts.TEMPLATES)}"
            )
        return key

    def _build_prompt(self, metric: str, concept, context, descriptors) -> str:
        base, desc_template = prompts.TEMPLATES[metric]
        if descriptors is not None:
            if desc_template is None:
                raise ValueError(f"metric {metric!r} has no descriptor prompt variant")
            return desc_template.format(
                prompt=concept, country=context, descriptors=descriptors
            )
        return base.format(prompt=concept, country=context)

    def score(
        self,
        image: str,
        metric: str,
        concept: str,
        context: str = "",
        descriptors: Optional[object] = None,
    ) -> float:
        metric = self._resolve_metric(metric)
        prompt = self._build_prompt(metric, concept, context, descriptors)

        scores: List[float] = []
        for _ in range(self.n_samples):
            response = self.mllm.generate(
                user_prompt=prompt,
                image_path=image,
                system_prompt=None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            s = parse_score(response)
            if s is None:
                continue
            if self.clamp:
                s = max(1.0, min(5.0, s))
            scores.append(s)

        if not scores:
            return math.nan
        return sum(scores) / len(scores)

    def align(self, image, concept, context="", descriptors=None) -> float:
        return self.score(image, "align", concept, context, descriptors)

    def hal(self, image, concept, context="", descriptors=None) -> float:
        return self.score(image, "hal", concept, context, descriptors)

    def exag(self, image, concept, context="") -> float:
        return self.score(image, "exag", concept, context)

    def evaluate(
        self,
        image: str,
        concept: str,
        context: str = "",
        metrics: Sequence[str] = ("align", "hal", "exag"),
        descriptors: Optional[object] = None,
    ) -> dict:
        out = {}
        for m in metrics:
            key = self._resolve_metric(m)
            desc = descriptors if key != "exag" else None
            out[key] = self.score(image, key, concept, context, desc)
        return out
