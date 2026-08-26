"""Gemini 2.5 Flash Image. Uses ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``."""

from __future__ import annotations

from io import BytesIO
from typing import List, Sequence

from PIL import Image

from ahead.registry import t2i
from ahead.cultivatebench.t2i.base import BaseT2I

MODEL_NAME = "gemini-2.5-flash-image-preview"


@t2i.register(MODEL_NAME)
class GeminiImage(BaseT2I):
    name = MODEL_NAME

    def __init__(self, model: str = MODEL_NAME):
        self.model = model
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client()  # reads GOOGLE_API_KEY / GEMINI_API_KEY from env
        return self._client

    def generate(
        self,
        prompts: Sequence[str],
        num_images: int = 1,
        size: str = "1024x1024",
        seed: int = 42,
    ) -> List:
        images = []
        for i in range(num_images):
            response = self._lazy_client().models.generate_content(
                model=self.model, contents=[prompts[i]]
            )
            candidate = response.candidates[0] if response.candidates else None
            parts = candidate.content.parts if candidate and candidate.content else []
            found = None
            for part in parts:
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None)
                if data:
                    found = Image.open(BytesIO(data))
                    break
            if found is None:
                raise RuntimeError(f"Gemini returned no image for prompt index {i}")
            images.append(found)
        return images
