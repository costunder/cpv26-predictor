"""Name-checked restoration of the project's single-group AdamW optimizer.

PyTorch optimizer checkpoints associate state with parameter positions, not
names. Recreate the saved order explicitly before loading: model registration
order can differ across processes even when all parameter names/shapes match.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cpv26.models._torch import require_torch


def _model_parameters(model: Any) -> dict[str, Any]:
    entries = list(model.named_parameters())
    if not entries:
        raise ValueError("AdamW requires model parameters")
    names = [name for name, _ in entries]
    identities = [id(parameter) for _, parameter in entries]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("model parameter names must be non-empty strings")
    if len(set(names)) != len(names) or len(set(identities)) != len(identities):
        raise ValueError("model parameter names and identities must be one-to-one")
    return dict(entries)


def _validate_names(names: Any, parameters: Mapping[str, Any], count: int) -> list[str]:
    if not isinstance(names, (list, tuple)) or any(not isinstance(name, str) for name in names):
        raise ValueError("optimizer parameter names must be a list of strings")
    result = list(names)
    if len(set(result)) != len(result):
        raise ValueError("duplicate optimizer parameter names")
    unknown = sorted(set(result) - set(parameters))
    if unknown:
        raise ValueError(f"unknown optimizer parameter names: {unknown}")
    if len(result) != count or set(result) != set(parameters):
        raise ValueError("optimizer parameter names/count must cover the entire model exactly once")
    return result


def optimizer_parameter_names(model: Any, optimizer: Any) -> list[list[str]]:
    """Return the actual optimizer order, rejecting partial/multi-group optimizers."""
    torch, _ = require_torch()
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("only the project's single-group AdamW optimizer is supported")
    if len(optimizer.param_groups) != 1:
        raise ValueError("only single-group AdamW optimizer checkpoints are supported")
    parameters = _model_parameters(model)
    by_identity = {id(parameter): name for name, parameter in parameters.items()}
    names: list[str] = []
    for parameter in optimizer.param_groups[0]["params"]:
        if id(parameter) not in by_identity:
            raise ValueError("optimizer contains a parameter outside the model")
        names.append(by_identity[id(parameter)])
    return [_validate_names(names, parameters, len(parameters))]


def _checkpoint_order(
    model: Any, parameters: Mapping[str, Any], checkpoint: Mapping[str, Any],
) -> tuple[list[str], Mapping[str, Any]]:
    saved = checkpoint.get("optimizer")
    if not isinstance(saved, Mapping):
        raise ValueError("checkpoint optimizer state must be a mapping")
    groups = saved.get("param_groups")
    if not isinstance(groups, (list, tuple)) or len(groups) != 1:
        raise ValueError("only single-group AdamW optimizer checkpoints are supported")
    if not isinstance(groups[0], Mapping):
        raise ValueError("optimizer parameter group must be a mapping")
    identifiers = groups[0].get("params")
    if not isinstance(identifiers, (list, tuple)) or any(
        isinstance(identifier, bool) or not isinstance(identifier, int)
        for identifier in identifiers
    ):
        raise ValueError("optimizer parameter IDs must be a list of integers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate optimizer parameter IDs")
    state = saved.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("optimizer state must be a mapping")
    if any(
        isinstance(identifier, bool) or not isinstance(identifier, int)
        or identifier not in identifiers for identifier in state
    ):
        raise ValueError("optimizer state contains unknown parameter IDs")

    if "optimizer_parameter_names" in checkpoint:
        metadata = checkpoint["optimizer_parameter_names"]
        if not isinstance(metadata, (list, tuple)) or len(metadata) != 1:
            raise ValueError("optimizer_parameter_names must describe exactly one parameter group")
        names = _validate_names(metadata[0], parameters, len(identifiers))
    else:
        # Legacy runner constructed AdamW(model.parameters()). state_dict keeps
        # that parameter traversal order, with buffers interspersed. Never infer
        # from the NEW model's registration order or from tensor shapes alone.
        saved_model = checkpoint.get("model")
        if not isinstance(saved_model, Mapping):
            raise ValueError("legacy optimizer restoration requires the saved model state_dict")
        current_keys = set(model.state_dict())
        if any(not isinstance(name, str) for name in saved_model):
            raise ValueError("legacy model state_dict keys must be strings")
        unknown = sorted(set(saved_model) - current_keys)
        if unknown:
            raise ValueError(f"legacy model state_dict contains unknown names: {unknown}")
        names = _validate_names(
            [name for name in saved_model if name in parameters], parameters, len(identifiers),
        )

    # load_state_dict itself accepts mismatched moment shapes and can fail much
    # later in step(). Reject those before an optimizer is exposed to the caller.
    torch, _ = require_torch()
    for identifier, name in zip(identifiers, names, strict=True):
        parameter_state = state.get(identifier, {})
        if not isinstance(parameter_state, Mapping):
            raise ValueError(f"optimizer state for {name} must be a mapping")
        for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            moment = parameter_state.get(field)
            if moment is not None and (
                not torch.is_tensor(moment) or moment.shape != parameters[name].shape
            ):
                raise ValueError(f"optimizer {field} shape does not match parameter {name}")
    return names, saved


def _clone_tensor_tree(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, Mapping):
        return {key: _clone_tensor_tree(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensor_tree(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item, torch) for item in value)
    return value


def make_adamw(
    model: Any, *, learning_rate: float, weight_decay: float,
    checkpoint: Mapping[str, Any] | None = None, clone_state: bool = False,
) -> Any:
    """Build/restore single-group AdamW using a verified saved parameter order.

    Saved optimizer hyperparameters retain their original load_state_dict
    behavior. ``clone_state=True`` also isolates every restored tensor state from
    the input checkpoint, including CPU step counters and nested tensor values.
    Model weights are not loaded or modified here; the caller owns model loading.
    """
    torch, _ = require_torch()
    parameters = _model_parameters(model)
    if checkpoint is None:
        names, saved = list(parameters), None
    else:
        names, saved = _checkpoint_order(model, parameters, checkpoint)
    optimizer = torch.optim.AdamW(
        [parameters[name] for name in names], lr=learning_rate, weight_decay=weight_decay,
    )
    if saved is not None:
        optimizer.load_state_dict(saved)
    if clone_state:
        for parameter, state in optimizer.state.items():
            optimizer.state[parameter] = _clone_tensor_tree(state, torch)
    return optimizer
