"""Shared helpers for metric ``.score`` inputs (mode, preds, refs, images)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from ahead.types import DescriptorSet

ImagePath = Union[str, Path]
ExtractFn = Callable[[Sequence[str], str], List[DescriptorSet]]

SINGLE = "single-image"
MULTI = "multi-image"


def as_mode(mode: Optional[str], default: str = SINGLE) -> str:
    m = (mode or default).lower().replace("_", "-")
    if m in ("single", "single-image", "singleimage"):
        return SINGLE
    if m in ("multi", "multi-image", "multiimage"):
        return MULTI
    raise ValueError(f"mode must be {SINGLE!r} or {MULTI!r}, got {mode!r}")


def is_descriptor_set(obj) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        return True
    if isinstance(obj, (str, bytes)):
        return False
    if isinstance(obj, Sequence):
        if len(obj) == 0:
            return True
        return isinstance(obj[0], str)
    return False


def resolve_preds(
    preds: Optional[Sequence[DescriptorSet]],
    images: Optional[Sequence[ImagePath]],
    concept: Optional[str],
    extract: Optional[ExtractFn],
) -> List[DescriptorSet]:
    if preds is not None:
        if is_descriptor_set(preds):
            raise ValueError(
                "preds must be a list of per-image descriptor sets, "
                'e.g. [{"objects": [...]}, {"objects": [...]}]'
            )
        return list(preds)
    if images is None:
        raise ValueError("Pass preds=[...] or images=[path, ...] with concept=...")
    if isinstance(images, (str, Path)):
        raise ValueError(
            "images must be a list of file paths, e.g. ['a.png', 'b.png']"
        )
    paths = [str(p) for p in images]
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise ValueError(f"image path(s) not found: {missing[:3]}")
    if concept is None:
        raise ValueError("concept= is required when extracting from images=")
    if extract is None:
        raise ValueError("extractor is not configured; pass extractor= at AheadMetrics init")
    return extract(paths, concept)


def resolve_refs_single(refs, n: int) -> List[DescriptorSet]:
    if is_descriptor_set(refs):
        raise ValueError(
            f"mode={SINGLE!r} requires refs as a list of length {n} "
            "(one reference descriptor set per image/pred)"
        )
    refs_list = list(refs)
    if len(refs_list) != n:
        raise ValueError(
            f"mode={SINGLE!r}: len(refs)={len(refs_list)} must equal "
            f"len(preds/images)={n}"
        )
    return refs_list


def resolve_refs_multi(refs) -> DescriptorSet:
    if not is_descriptor_set(refs):
        if isinstance(refs, Sequence) and len(refs) == 1 and is_descriptor_set(refs[0]):
            return refs[0]
        raise ValueError(
            f"mode={MULTI!r} requires refs as one descriptor set "
            "{dimension: [descriptors, ...]}"
        )
    return refs
