"""Resolve files/dirs/globs to sorted image paths."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

Source = Union[str, Path, Iterable["Source"], None]


def collect_image_paths(source: Source) -> List[str]:
    if source is None:
        return []
    if isinstance(source, (list, tuple, set)):
        out: List[str] = []
        for item in source:
            out.extend(collect_image_paths(item))
        return out
    text = str(source)
    if any(ch in text for ch in "*?["):
        return sorted(
            str(p)
            for p in Path().glob(text)
            if p.suffix.lower() in IMAGE_EXTS and p.is_file()
        )
    path = Path(text)
    if path.is_dir():
        return sorted(
            str(p)
            for p in path.iterdir()
            if p.suffix.lower() in IMAGE_EXTS and p.is_file()
        )
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [str(path)]
    return []
