# Contributing / extending AHEAD

Registries let you add backends without touching metric code.

## Add an image-text scorer (EXAG)

```python
from ahead.registry import scorers

@scorers.register("myscorer")
class MyScorer:
    def __init__(self, device="cuda"): ...
    def score(self, images, texts):        # -> tensor [n_images, n_texts]
        ...
```

Then: `AheadMetrics(scorer=scorers.create("myscorer"))`.

## Add a matcher (ALIGN / HAL / DDIV / SDIV)

```python
from ahead.registry import matchers
from ahead.metrics import AheadMetrics

@matchers.register("mymatcher")
class MyMatcher:
    def similarity(self, pred, ref, image=None):   # -> tensor [n_pred, n_ref] in [0,1]
        ...

AheadMetrics(matcher="mymatcher")
```

## Add a text-to-image generator (CULTIVateBench)

```python
from ahead.registry import t2i
from ahead.cultivatebench.t2i.base import BaseT2I
from ahead.cultivatebench import BenchmarkGenerator

@t2i.register("my-t2i")
class MyT2I(BaseT2I):
    name = "my-t2i"
    def generate(self, prompts, num_images=1, size="1024x1024", seed=42):
        ...

BenchmarkGenerator(t2i="my-t2i")
```

## Add an MLLM backbone (extraction / judge)

```python
from ahead.registry import mllms
from ahead.backbones.mllm import BaseMLLM

@mllms.register("my-vlm")
class MyVLM(BaseMLLM):
    def generate(self, user_prompt, image_path=None, system_prompt=None,
                 temperature=0.0, max_tokens=256) -> str:
        ...
```
