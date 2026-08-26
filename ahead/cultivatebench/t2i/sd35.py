"""Stable Diffusion 3.5 Medium."""

from __future__ import annotations

from typing import List, Sequence

from ahead.registry import t2i
from ahead.cultivatebench.t2i.base import BaseT2I, torch_seed_generator


@t2i.register("stable-diffusion-3.5-medium")
class SD35(BaseT2I):
    name = "stable-diffusion-3.5-medium"

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3.5-medium",
        device: str = "cuda",
        num_inference_steps: int = 50,
    ):
        import torch
        from diffusers import StableDiffusion3Pipeline

        self.device = device
        self.num_inference_steps = num_inference_steps
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
        ).to(device)
        if device == "cuda":
            self.pipe.enable_model_cpu_offload()
        self.negative_prompt = (
            "distorted, deformed, disfigured, bad anatomy, extra limbs, "
            "blurry, low quality, low resolution, watermark, text, logo, "
            "duplicate, poorly drawn, mutated, mutilated"
        )

    def generate(
        self,
        prompts: Sequence[str],
        num_images: int = 1,
        size: str = "1024x1024",
        seed: int = 42,
    ) -> List:
        width, height = map(int, size.split("x"))
        images = []
        for i in range(num_images):
            out = self.pipe(
                prompt=prompts[i],
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=1,
                num_inference_steps=self.num_inference_steps,
                width=width,
                height=height,
                generator=torch_seed_generator(self.device, seed, i),
            )
            images.append(out.images[0])
        return images
