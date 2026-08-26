"""MLLM-as-a-judge baseline (single-image, 1–5 Likert)."""

from ahead.metrics.mllm_judge.judge import MLLMJudge, parse_score

__all__ = ["MLLMJudge", "parse_score"]
