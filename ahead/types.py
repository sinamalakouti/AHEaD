"""Shared types: descriptor sets and metric results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Union

DescriptorSet = Union[Mapping[str, Sequence[str]], Sequence[str]]

DEFAULT_DIMENSION = "_"


def normalise_descriptors(descriptors: DescriptorSet) -> Dict[str, List[str]]:
    if descriptors is None:
        return {}
    if isinstance(descriptors, Mapping):
        return {str(dim): list(vals) for dim, vals in descriptors.items()}
    return {DEFAULT_DIMENSION: list(descriptors)}


@dataclass
class MetricResult:
    name: str
    value: float
    per_dimension: Dict[str, float] = field(default_factory=dict)
    per_descriptor: Dict[str, float] = field(default_factory=dict)
    per_descriptor_by_dimension: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {self.name: self.value}
        if self.per_dimension:
            out[f"{self.name}_per_dimension"] = self.per_dimension
        if self.per_descriptor:
            out[f"{self.name}_per_descriptor"] = self.per_descriptor
        if self.per_descriptor_by_dimension:
            out[f"{self.name}_per_descriptor_by_dimension"] = (
                self.per_descriptor_by_dimension
            )
        return out


@dataclass
class AheadReport:
    metrics: Dict[str, MetricResult] = field(default_factory=dict)

    def add(self, result: MetricResult) -> "AheadReport":
        self.metrics[result.name] = result
        return self

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for result in self.metrics.values():
            out.update(result.to_dict())
        return out
