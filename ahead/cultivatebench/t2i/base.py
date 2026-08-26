from __future__ import annotations
import torch

from abc import ABC, abstractmethod
from typing import List, Sequence

from ahead.registry import t2i as t2i_registry

PROMPT_TEMPLATE = "A photorealistic photo of {activity} in {country}."

# Registry keys are model names; aliases resolve case-insensitively.
_ALIASES = {
    # SD-3.5-medium
    "stable-diffusion-3.5-medium": "stable-diffusion-3.5-medium",
    "sd3.5": "stable-diffusion-3.5-medium",
    "sd-3.5": "stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-medium": "stable-diffusion-3.5-medium",
    # FLUX.1-dev
    "flux.1-dev": "FLUX.1-dev",
    "flux": "FLUX.1-dev",
    "flux1": "FLUX.1-dev",
    "black-forest-labs/flux.1-dev": "FLUX.1-dev",
    # Qwen-Image
    "qwen-image": "Qwen-Image",
    "qwen/qwen-image": "Qwen-Image",
    # OpenAI Images
    "dall-e-3": "dall-e-3",
    "dalle3": "dall-e-3",
    "gpt-image-1": "gpt-image-1",
    # Gemini image
    "gemini-2.5-flash-image-preview": "gemini-2.5-flash-image-preview",
    "gemini": "gemini-2.5-flash-image-preview",
    "nano-banana": "gemini-2.5-flash-image-preview",
    "gemini-2.5-flash-image": "gemini-2.5-flash-image-preview",
}

PUBLIC_MODELS = (
    "stable-diffusion-3.5-medium",
    "FLUX.1-dev",
    "Qwen-Image",
)
PROPRIETARY_MODELS = (
    "dall-e-3",
    "gpt-image-1",
    "gemini-2.5-flash-image-preview",
)


def _resolve_name(name: str) -> str:
    return _ALIASES.get(name.lower(), name)


def make_prompt(activity: str, country: str, template: str = PROMPT_TEMPLATE) -> str:
    return template.format(activity=activity, country=country)


def default_num_images(model: str) -> int:
    """10 for public models, 1 for proprietary."""
    key = _resolve_name(model)
    return 10 if key in PUBLIC_MODELS else 1


class BaseT2I(ABC):
    name: str = "t2i"

    @abstractmethod
    def generate(
        self,
        prompts: Sequence[str],
        num_images: int = 1,
        size: str = "1024x1024",
        seed: int = 42,
    ) -> List["object"]:
        raise NotImplementedError


def _ensure_registered() -> None:
    from ahead.cultivatebench.t2i import (
        dalle3,
        flux,
        gemini,
        gpt_image,
        qwen,
        sd35,
    )  # noqa: F401


def get_t2i(name: str, **kwargs) -> BaseT2I:
    _ensure_registered()
    return t2i_registry.create(_resolve_name(name), **kwargs)


def torch_seed_generator(device: str, seed: int, offset: int):
    gen = torch.Generator(device=device if device != "cpu" else "cpu")
    if seed > 0:
        gen.manual_seed(seed + offset)
    return gen
