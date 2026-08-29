"""Optional PyTorch dependency boundary for model modules."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any


class TorchUnavailableError(RuntimeError):
    """Raised when a neural model is used without the optional ML runtime."""


torch: Any
nn: Any
try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
except ImportError:
    torch = None
    nn = None

if TYPE_CHECKING:

    class ModuleBase:
        """Static shape of the small ``nn.Module`` surface used in this package."""

        training: bool

        def __init__(self) -> None:
            self.training = False

        def parameters(self) -> Iterator[Any]:
            return iter(())

else:
    ModuleBase = nn.Module if nn is not None else object


def torch_available() -> bool:
    return torch is not None


def require_torch() -> tuple[Any, Any]:
    if torch is None or nn is None:
        raise TorchUnavailableError(
            "PyTorch is required for cpv26 neural models. Install the project's "
            "ML dependency group in the Linux runtime before constructing this class."
        )
    return torch, nn
