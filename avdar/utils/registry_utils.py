import logging
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


class Registry(Generic[T]):
    """OOP factory magic"""

    def __init__(self, name: str, base: Callable[..., T]) -> None:
        self.name = name
        self.constructors: dict[str, Callable[..., T]] = {}
        self.shared_keys: dict[str, Any] = {}

    def add(self, name: str, constructor: Callable[..., T], shared_keys = []) -> None:
        logger.info(f"Register {self.name}: {name}")
        self.constructors[name] = constructor
        self.shared_keys[name] = shared_keys

    def get(self, name: str) -> Callable[..., T]:
        return self.constructors[name]

    def build(self, name: str, kwargs: dict[str, Any]) -> T:
        return self.get(name)(**kwargs)

    def build_shared(self, name: str, kwargs: dict[str, Any], shared_kwargs: dict[str, Any]) -> T:
        skwargs = {k: shared_kwargs[k] for k in self.shared_keys[name]}
        return self.get(name)(**skwargs, **kwargs)