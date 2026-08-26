"""Generate CULTIVate images from the CultureBench prompt catalog.

  python scripts/generate_benchmark.py --t2i FLUX.1-dev --catalog cultivate --out images/
  python scripts/generate_benchmark.py --t2i gpt-image-1 \\
      --catalog /path/to/culturebench_prompts.json --countries IRAN USA --out images/
"""

from __future__ import annotations

import argparse

from ahead.cultivatebench import BenchmarkGenerator, load_catalog
from ahead.cultivatebench.t2i.base import default_num_images


def parse_args():
    p = argparse.ArgumentParser(description="Generate CULTIVate / CultureBench images")
    p.add_argument(
        "--t2i",
        required=True,
        help=(
            "Model name: stable-diffusion-3.5-medium | FLUX.1-dev | Qwen-Image | "
            "dall-e-3 | gpt-image-1 | gemini-2.5-flash-image-preview "
            "(short aliases: sd3.5, flux, qwen-image, gemini)"
        ),
    )
    p.add_argument(
        "--catalog",
        default="cultivate",
        help="Local JSON path, URL, or alias 'cultivate'",
    )
    p.add_argument("--out", required=True, help="Output root directory")
    p.add_argument("--countries", nargs="+", default=None)
    p.add_argument("--num-images", type=int, default=None, help="Default: 10 public / 1 proprietary")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", default="1024x1024")
    return p.parse_args()


def main():
    args = parse_args()
    items = load_catalog(args.catalog, countries=args.countries)
    if not items:
        raise SystemExit(f"no prompts loaded from catalog={args.catalog!r}")
    n = args.num_images if args.num_images is not None else default_num_images(args.t2i)
    gen = BenchmarkGenerator(t2i=args.t2i, num_images=n, seed=args.seed, size=args.size)
    written = gen.generate(items, args.out)
    print(f"wrote {len(written)} images for {len(items)} prompts")


if __name__ == "__main__":
    main()
