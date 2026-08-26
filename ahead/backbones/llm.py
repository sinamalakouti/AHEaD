"""Text-only LLM backbone for reference descriptors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ahead.registry import llms


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        raise NotImplementedError


@llms.register("gpt-4o")
class OpenAILLM(BaseLLM):
    """Uses ``OPENAI_API_KEY``."""

    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__(model_name)
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI()
        return self._client

    def generate(self, user_prompt, system_prompt=None, temperature=0.0, max_tokens=512) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        resp = self._lazy_client().chat.completions.create(
            model=self.model_name, messages=messages, temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


@llms.register("gemini-2.5-flash")
class GeminiLLM(BaseLLM):
    """Uses ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        super().__init__(model_name)
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    def generate(self, user_prompt, system_prompt=None, temperature=0.0, max_tokens=512) -> str:
        from google.genai import types

        config_kwargs = {"temperature": temperature, "max_output_tokens": max_tokens}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        response = self._lazy_client().models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return (response.text or "").strip()


def get_llm(name: str = "gpt-4o", **kwargs) -> BaseLLM:
    return llms.create(name, **kwargs)
