"""Generate CULTIVate images from activity–country prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from ahead.cultivatebench.t2i.base import (
    BaseT2I,
    default_num_images,
    get_t2i,
    make_prompt,
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "item"


@dataclass
class PromptItem:
    country: str
    activity: str
    subactivity: str = ""
    text: str = ""

    def phrase(self) -> str:
        return self.text or self.subactivity or self.activity

    def prompt(self) -> str:
        return make_prompt(self.phrase(), self.country)


# CULTIVate prompt catalog:
# https://github.com/sinamalakouti/AHEaD/tree/master/data/culturebench
CULTIVATE_PROMPTS_URL = (
    "https://raw.githubusercontent.com/sinamalakouti/AHEaD/master/"
    "data/culturebench/culturebench_prompts.json"
)


def _read_catalog_json(source: Union[str, Path]) -> dict:
    text = str(source).strip()
    if text.lower() in {"cultivate", "culturebench", "ahead"}:
        text = CULTIVATE_PROMPTS_URL
    if text.startswith("http://") or text.startswith("https://"):
        import urllib.request

        with urllib.request.urlopen(text) as resp:  # nosec - user-provided catalog URL
            return json.loads(resp.read().decode("utf-8"))
    return json.loads(Path(text).read_text())


def load_catalog(
    path: Union[str, Path] = "cultivate",
    countries: Optional[Sequence[str]] = None,
) -> List[PromptItem]:
    """Load ``{country: {activity: {subactivity: phrase}}}`` from a path, URL, or ``\"cultivate\"``."""
    data = _read_catalog_json(path)
    keep = {c.upper() for c in countries} if countries else None
    items: List[PromptItem] = []
    for country, activities in data.items():
        if keep is not None and country.upper() not in keep:
            continue
        if not isinstance(activities, dict):
            continue
        for activity, subs in activities.items():
            if isinstance(subs, dict):
                for sub, phrase in subs.items():
                    if not isinstance(phrase, str):
                        continue
                    items.append(
                        PromptItem(
                            country=country,
                            activity=str(activity),
                            subactivity=str(sub),
                            text=phrase,
                        )
                    )
            elif isinstance(subs, str):
                items.append(PromptItem(country=country, activity=str(activity), text=subs))
            elif isinstance(subs, list):
                for sub in subs:
                    items.append(
                        PromptItem(
                            country=country,
                            activity=str(activity),
                            subactivity=str(sub),
                            text=str(sub),
                        )
                    )
    return items


class BenchmarkGenerator:
    def __init__(
        self,
        t2i: "BaseT2I | str",
        num_images: Optional[int] = None,
        seed: int = 42,
        size: str = "1024x1024",
        **t2i_kwargs,
    ):
        self.generator = t2i if isinstance(t2i, BaseT2I) else get_t2i(t2i, **t2i_kwargs)
        name = getattr(self.generator, "name", str(t2i))
        self.num_images = num_images if num_images is not None else default_num_images(name)
        self.seed = seed
        self.size = size

    def output_dir(self, root: Union[str, Path], item: PromptItem) -> Path:
        parts = [Path(root), _slug(self.generator.name), _slug(item.country), _slug(item.activity)]
        if item.subactivity:
            parts.append(_slug(item.subactivity))
        return Path(*parts)

    def generate(
        self,
        items: Iterable[PromptItem],
        output_dir: Union[str, Path],
        skip_existing: bool = True,
    ) -> List[Path]:
        written: List[Path] = []
        root = Path(output_dir)
        for item in items:
            dest = self.output_dir(root, item)
            dest.mkdir(parents=True, exist_ok=True)
            existing = sorted(dest.glob("*.png"))
            if skip_existing and len(existing) >= self.num_images:
                continue
            n_needed = self.num_images - len(existing)
            offset = len(existing)
            prompt = item.prompt()
            prompts = [prompt] * n_needed
            images = self.generator.generate(
                prompts, num_images=n_needed, size=self.size, seed=self.seed + offset
            )
            from PIL import Image

            for k, img in enumerate(images):
                if not isinstance(img, Image.Image):
                    continue
                img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
                path = dest / f"{offset + k:03d}.png"
                img.save(path)
                written.append(path)
            (dest / "prompt.txt").write_text(prompt + "\n")
        return written
