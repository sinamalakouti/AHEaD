"""Multimodal LLM backbone (descriptor extraction / MLLM judge)."""

from __future__ import annotations

import base64
import mimetypes
from abc import ABC, abstractmethod
from typing import Optional

from ahead.registry import mllms


class BaseMLLM(ABC):
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        raise NotImplementedError


@mllms.register("gpt-4o")
class OpenAIMLLM(BaseMLLM):
    """Uses ``OPENAI_API_KEY``."""

    def __init__(self, model_name: str = "gpt-4o", device: str = "cuda"):
        super().__init__(model_name, device)
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI()
        return self._client

    @staticmethod
    def _image_data_url(image_path: str) -> str:
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{encoded}"

    def generate(self, user_prompt, image_path=None, system_prompt=None,
                 temperature=0.0, max_tokens=256) -> str:
        content = [{"type": "text", "text": user_prompt}]
        if image_path is not None:
            content.append(
                {"type": "image_url", "image_url": {"url": self._image_data_url(image_path)}}
            )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        resp = self._lazy_client().chat.completions.create(
            model=self.model_name, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


class _UnregisteredMLLM(BaseMLLM):
    def generate(self, *args, **kwargs) -> str:
        raise NotImplementedError(
            f"No loader is registered for MLLM backbone {self.model_name!r}. "
            f"Register one with @mllms.register({self.model_name!r}) (see CONTRIBUTING.md)."
        )


for _name in ("llava", "qwen", "internvl"):
    mllms.register(_name, lambda name=_name, device="cuda": _UnregisteredMLLM(name, device))


def get_mllm(name: str = "internvl", device: str = "cuda") -> BaseMLLM:
    return mllms.create(name, device=device)
