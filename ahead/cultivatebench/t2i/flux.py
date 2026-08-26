"""FLUX.1-dev."""

from __future__ import annotations

from typing import List, Sequence

from ahead.registry import t2i
from ahead.cultivatebench.t2i.base import BaseT2I, torch_seed_generator


@t2i.register("FLUX.1-dev")
class FLUX(BaseT2I):
    name = "FLUX.1-dev"

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda",
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
    ):
        import torch
        from diffusers import FluxPipeline

        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
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
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                max_sequence_length=512,
                width=width,
                height=height,
                generator=torch_seed_generator(self.device, seed, i),
            )
            images.append(out.images[0])
        return images
