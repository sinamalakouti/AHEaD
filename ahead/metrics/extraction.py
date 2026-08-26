"""MLLM descriptor extraction from images."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from ahead.backbones.mllm import BaseMLLM, get_mllm

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise visual analyst. List only concrete, visible elements. "
    "Return a comma-separated list of at most {max_items} short descriptors."
)

DEFAULT_AXIS_TEMPLATES: Dict[str, str] = {
    "setting": "List the setting/scene elements visible for '{concept}'.",
    "objects": "List the distinctive objects/items visible for '{concept}'.",
    "attire": "List the clothing/attire details visible for '{concept}'.",
    "interaction": "List how people interact/what actions occur for '{concept}'.",
    "spatial": "List the spatial arrangement/composition for '{concept}'.",
}


def _parse_list(raw: str) -> List[str]:
    raw = (raw or "").split("assistant:")[-1]
    return [d.strip() for d in raw.split(",") if d.strip()]


class DescriptorExtractor:
    def __init__(
        self,
        mllm: "BaseMLLM | str" = "internvl",
        axis_templates: Optional[Mapping[str, str]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_items: int = 8,
        device: str = "cuda",
    ):
        self.mllm = (
            mllm if isinstance(mllm, BaseMLLM) else get_mllm(mllm, device=device)
        )
        self.axis_templates = dict(axis_templates or DEFAULT_AXIS_TEMPLATES)
        self.system_prompt = system_prompt
        self.max_items = max_items

    def extract(self, image_path: str, concept: str) -> Dict[str, List[str]]:
        system = self.system_prompt.format(max_items=self.max_items)
        out: Dict[str, List[str]] = {}
        for dim, template in self.axis_templates.items():
            raw = self.mllm.generate(
                user_prompt=template.format(concept=concept, max_items=self.max_items),
                image_path=image_path,
                system_prompt=system,
                temperature=0.0,
                max_tokens=64,
            )
            out[dim] = _parse_list(raw)
        return out

    def extract_batch(self, image_paths, concept: str) -> List[Dict[str, List[str]]]:
        return [self.extract(p, concept) for p in image_paths]
