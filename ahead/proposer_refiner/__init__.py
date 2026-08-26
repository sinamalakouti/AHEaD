"""Reference descriptors: propose → union → refine."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

from ahead.backbones.llm import BaseLLM
from ahead.proposer_refiner.proposer import LLMDescriptorProposer, union_descriptors
from ahead.proposer_refiner.refiner import LLMDescriptorRefiner, country_to_country_prompt

PAPER_PROPOSERS = ("gemini-2.5-flash", "gpt-4o")
PAPER_REFINER = "gpt-4o"


def descriptor_tokens(raw: Dict[str, List]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for dim, items in raw.items():
        tokens = []
        for it in items:
            if isinstance(it, dict) and it.get("token"):
                tokens.append(str(it["token"]).strip())
            elif isinstance(it, str) and it.strip():
                tokens.append(it.strip())
        out[dim] = tokens
    return out


def activity_prompt(concept: str, country: str = "") -> str:
    if country and country not in concept:
        return f"{concept} in {country_to_country_prompt(country)}"
    return concept


class ProposerRefiner:
    def __init__(
        self,
        proposers: Sequence[Union[str, BaseLLM]] = PAPER_PROPOSERS,
        refiner: Union[str, BaseLLM] = PAPER_REFINER,
        proposer: Optional[LLMDescriptorProposer] = None,
        refiner_model: Optional[LLMDescriptorRefiner] = None,
    ):
        self.proposers = list(proposers)
        self.refiner_llm = refiner
        self.proposer = proposer or LLMDescriptorProposer()
        self.refiner = refiner_model or LLMDescriptorRefiner()

    def propose(self, concept: str, country: str = "") -> Dict[str, List[dict]]:
        prompt = activity_prompt(concept, country)
        proposals = [self.proposer.propose(llm, prompt) for llm in self.proposers]
        return union_descriptors(proposals) if len(proposals) > 1 else proposals[0]

    def refine(
        self, candidates: Dict[str, List[dict]], concept: str, country: str = ""
    ) -> Dict[str, List[dict]]:
        prompt = concept
        return self.refiner.refine(self.refiner_llm, candidates, prompt, country)

    def generate(
        self, concept: str, country: str = "", refine: bool = True, as_tokens: bool = True
    ) -> dict:
        candidates = self.propose(concept, country)
        out = self.refine(candidates, concept, country) if refine else candidates
        return descriptor_tokens(out) if as_tokens else out


__all__ = [
    "PAPER_PROPOSERS",
    "PAPER_REFINER",
    "LLMDescriptorProposer",
    "LLMDescriptorRefiner",
    "ProposerRefiner",
    "union_descriptors",
    "descriptor_tokens",
    "activity_prompt",
]
