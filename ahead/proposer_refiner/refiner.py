"""LLM refiner: clean union candidates per dimension."""

from __future__ import annotations

from typing import Dict, List

from ahead.backbones.llm import BaseLLM, get_llm
from ahead.proposer_refiner.prompts import refiner_prompt_template
from ahead.proposer_refiner.proposer import RETRY_JSON, parse_descriptor_json


def country_to_country_prompt(country: str) -> str:
    return country.replace("_", " ").lower().capitalize()


class LLMDescriptorRefiner:
    def __init__(
        self,
        temperature: float = 0.1,
        num_max_descriptors: int = 10,
        max_retries: int = 3,
    ):
        self.prompt_template = refiner_prompt_template
        self.temperature = temperature
        self.num_max_descriptors = num_max_descriptors
        self.max_retries = max_retries

    def refine(
        self,
        llm: "BaseLLM | str",
        descriptor_candidate: Dict[str, List[dict]],
        prompt: str,
        country: str,
    ) -> Dict[str, List[dict]]:
        llm = llm if isinstance(llm, BaseLLM) else get_llm(llm)
        refined: Dict[str, List[dict]] = {}
        country_prompt = country_to_country_prompt(country) if country else country
        for dimension, candidates in descriptor_candidate.items():
            user_prompt = self.prompt_template.format(
                prompt=prompt,
                country=country_prompt,
                dimension=dimension,
                candidate_descriptors=candidates,
                max_items=self.num_max_descriptors,
            )
            result: List[dict] = []
            for attempt in range(self.max_retries):
                response = llm.generate(
                    user_prompt=user_prompt,
                    system_prompt=None,
                    temperature=self.temperature,
                    max_tokens=2048,
                )
                parsed = parse_descriptor_json(response)
                if parsed or (response or "").strip() in ("[]", "{}", '{"descriptors":[]}'):
                    result = parsed
                    break
                user_prompt = user_prompt + "\n\n" + RETRY_JSON
            refined[dimension] = result
        return refined
