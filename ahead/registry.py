"""Name → factory registry for extensible backends."""

from __future__ import annotations

from typing import Callable, Dict, Generic, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._factories: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T] | None = None):
        def _add(f: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise ValueError(f"{self.kind} {name!r} already registered")
            self._factories[name] = f
            return f

        return _add if factory is None else _add(factory)

    def create(self, name: str, **kwargs) -> T:
        if name not in self._factories:
            raise ValueError(
                f"unknown {self.kind} {name!r}; available: {self.available()}"
            )
        return self._factories[name](**kwargs)

    def available(self) -> List[str]:
        return sorted(self._factories)


matchers: "Registry" = Registry("matcher")
scorers: "Registry" = Registry("image-text scorer")
mllms: "Registry" = Registry("mllm backbone")
llms: "Registry" = Registry("llm backbone")
t2i: "Registry" = Registry("text-to-image generator")
