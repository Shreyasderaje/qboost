"""Dataset utilities: task registry, synthetic generators, real-data adapters."""

from backend.data.synth import (
    make_dataset,
    make_federated_splits,
)
from backend.data.tasks import TASKS, get_task

__all__ = ["TASKS", "get_task", "make_dataset", "make_federated_splits"]
