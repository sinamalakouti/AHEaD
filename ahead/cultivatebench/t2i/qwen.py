"""Qwen-Image (+ Lightning LoRA)."""

from __future__ import annotations

import math
from typing import List, Sequence

from ahead.registry import t2i
from ahead.cultivatebench.t2i.base import BaseT2I, torch_seed_generator


@t2i.register("Qwen-Image")
class QwenImage(BaseT2I):
    name = "Qwen-Image"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image",
        device: str = "cuda",
        distilled: bool = True,
    ):
        import torch
        from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler

        self.device = device
        self.distilled = distilled
        if distilled:
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                {
                    "base_image_seq_len": 256,
                    "base_shift": math.log(3),
                    "invert_sigmas": False,
                    "max_image_seq_len": 8192,
                    "max_shift": math.log(3),
                    "num_train_timesteps": 1000,
                    "shift": 1.0,
                    "shift_terminal": None,
                    "stochastic_sampling": False,
                    "time_shift_type": "exponential",
                    "use_beta_sigmas": False,
                    "use_dynamic_shifting": True,
                    "use_exponential_sigmas": False,
                    "use_karras_sigmas": False,
                }
            )
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                scheduler=scheduler,
            )
            self.pipe.load_lora_weights(
                "lightx2v/Qwen-Image-Lightning",
                weight_name="Qwen-Image-Lightning-8steps-V1.0.safetensors",
            )
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            )
        if device == "cuda":
            self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)
        self.negative_prompt = (
            "distorted, deformed, disfigured, bad anatomy, extra limbs, "
            "blurry, low quality, low resolution, watermark, text, logo, "
            "duplicate, poorly drawn, mutated, mutilated"
        )
        self.positive_magic = ", Ultra HD, 4K, cinematic composition."

    def generate(
        self,
        prompts: Sequence[str],
        num_images: int = 1,
        size: str = "1024x1024",
        seed: int = 42,
    ) -> List:
        width, height = map(int, size.split("x"))
        steps = 4 if self.distilled else 50
        cfg = 1.0 if self.distilled else 4.0
        images = []
        for i in range(num_images):
            out = self.pipe(
                prompt=prompts[i] + self.positive_magic,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=1,
                num_inference_steps=steps,
                true_cfg_scale=cfg,
                width=width,
                height=height,
                generator=torch_seed_generator(self.device, seed, i),
            )
            images.append(out.images[0])
        return images
