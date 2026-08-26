"""LLM proposer: per-dimension candidate descriptors."""

from __future__ import annotations

import json
from typing import Dict, List

from ahead.backbones.llm import BaseLLM, get_llm
from ahead.proposer_refiner.prompts import (
    PROMPT_ATTIRE_TEMPLATE,
    PROMPT_INTERACTION_TEMPLATE,
    PROMPT_OBJECTS_TEMPLATE,
    PROMPT_PEOPLE_TEMPLATE,
    PROMPT_SETTING_TEMPLATE,
    PROMPT_SPATIAL_TEMPLATE,
    SYSTEM_PROMPT,
)

RETRY_JSON = (
    'Previous response was not valid JSON. Please return ONLY valid JSON in this format: '
    '{"descriptors": [{"token": "example", "style": "traditional", "necessity": "necessary"}]}'
)


def _strip_fences(text: str) -> str:
    return (text or "").strip().replace("```json", "").replace("```", "").replace("json", "").strip()


def parse_descriptor_json(raw: str) -> List[dict]:
    text = _strip_fences(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(obj, dict):
        items = obj.get("descriptors", [])
    elif isinstance(obj, list):
        items = obj
    else:
        items = []
    return [it for it in items if it is not None]


class LLMDescriptorProposer:
    def __init__(
        self,
        temperature: float = 0.25,
        num_max_descriptors: int = 10,
        max_retries: int = 3,
    ):
        self.dimension_templates = {
            "setting": PROMPT_SETTING_TEMPLATE,
            "objects": PROMPT_OBJECTS_TEMPLATE,
            "attire": PROMPT_ATTIRE_TEMPLATE,
            "interaction": PROMPT_INTERACTION_TEMPLATE,
            "spatial": PROMPT_SPATIAL_TEMPLATE,
            "people": PROMPT_PEOPLE_TEMPLATE,
        }
        self.system_prompt = SYSTEM_PROMPT
        self.temperature = temperature
        self.num_max_descriptors = num_max_descriptors
        self.max_retries = max_retries

    def propose(self, llm: "BaseLLM | str", activity_prompt: str) -> Dict[str, List[dict]]:
        llm = llm if isinstance(llm, BaseLLM) else get_llm(llm)
        all_descriptors: Dict[str, List[dict]] = {}
        for dimension, template in self.dimension_templates.items():
            user_prompt = template.format(
                concept=activity_prompt, max_items=self.num_max_descriptors
            )
            system = self.system_prompt
            result: List[dict] = []
            for attempt in range(self.max_retries):
                response = llm.generate(
                    user_prompt=user_prompt,
                    system_prompt=system,
                    temperature=self.temperature,
                    max_tokens=2048,
                )
                parsed = parse_descriptor_json(response)
                if parsed or response.strip() in ("{}", "[]", '{"descriptors":[]}'):
                    result = parsed
                    break
                system = system + "\n\n" + RETRY_JSON
            all_descriptors[dimension] = result
        return all_descriptors


def union_descriptors(descriptors_list: List[Dict[str, List[dict]]]) -> Dict[str, List[dict]]:
    if not descriptors_list:
        return {}
    combined: Dict[str, List[dict]] = {}
    dimensions = list(descriptors_list[0].keys())
    for dimension in dimensions:
        combined[dimension] = [
            desc
            for descriptors in descriptors_list
            for desc in descriptors.get(dimension, [])
        ]
    return combined
