"""GPT Image 1 (OpenAI Images API). Uses ``OPENAI_API_KEY``."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import List, Sequence

from PIL import Image

from ahead.registry import t2i
from ahead.cultivatebench.t2i.base import BaseT2I


@t2i.register("gpt-image-1")
class GPTImage1(BaseT2I):
    name = "gpt-image-1"

    def __init__(self, model: str = "gpt-image-1"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def generate(
        self,
        prompts: Sequence[str],
        num_images: int = 1,
        size: str = "1024x1024",
        seed: int = 42,
    ) -> List:
        images = []
        for i in range(num_images):
            result = self.client.images.generate(
                model=self.model, prompt=prompts[i], size=size
            )
            raw = base64.b64decode(result.data[0].b64_json)
            images.append(Image.open(BytesIO(raw)))
        return images
