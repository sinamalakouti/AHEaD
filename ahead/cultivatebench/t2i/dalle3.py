"""DALL·E 3 (OpenAI Images API). Uses ``OPENAI_API_KEY``."""

from __future__ import annotations

from io import BytesIO
from typing import List, Sequence

import requests
from PIL import Image

from ahead.registry import t2i
from ahead.cultivatebench.t2i.base import BaseT2I


@t2i.register("dall-e-3")
class Dalle3(BaseT2I):
    name = "dall-e-3"

    def __init__(self, quality: str = "standard"):
        from openai import OpenAI

        self.client = OpenAI()
        self.quality = quality

    def generate(
        self,
        prompts: Sequence[str],
        num_images: int = 1,
        size: str = "1024x1024",
        seed: int = 42,
    ) -> List:
        images = []
        for i in range(num_images):
            resp = self.client.images.generate(
                model="dall-e-3",
                prompt=prompts[i],
                size=size,
                quality=self.quality,
                n=1,
            )
            url = resp.data[0].url
            raw = requests.get(url, timeout=60)
            raw.raise_for_status()
            images.append(Image.open(BytesIO(raw.content)))
        return images
