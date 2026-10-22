"""Optional observability; importing this package never constructs external clients."""

from prompt_learning.observability.phoenix import (
    DatasetExample, NoOpObserver, Observer, RunSpan, build_observer,
)

__all__ = ["DatasetExample", "NoOpObserver", "Observer", "RunSpan", "build_observer"]
