from __future__ import annotations

import copy
import os
import subprocess
import sys
import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest

from cpv26.training.optimizer_state import make_adamw, optimizer_parameter_names


@pytest.fixture
def torch() -> Any:
    return pytest.importorskip("torch")


def _model(torch: Any, order: tuple[str, ...] = ("batting", "pitching", "fielding")) -> Any:
    model = torch.nn.Module()
    model.register_buffer("buffer_before_roles", torch.tensor([99.0]))
    model.roles = torch.nn.ModuleDict()
    for index, name in enumerate(order):
        layer = torch.nn.Linear(2, 2, bias=False)
        layer.register_buffer("buffer_after_weight", torch.tensor([float(index)]))
        model.roles[name] = layer
    return model


def _checkpoint(torch: Any, *, named: bool = True) -> tuple[Any, dict[str, Any]]:
    model = _model(torch)
    optimizer = make_adamw(model, learning_rate=0.01, weight_decay=0.02)
    for index, (_, parameter) in enumerate(model.named_parameters()):
        parameter.grad = torch.full_like(parameter, float(index + 1))
    optimizer.step()
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    if named:
        checkpoint["optimizer_parameter_names"] = optimizer_parameter_names(model, optimizer)
    return model, checkpoint


def _assert_named_moments(torch: Any, optimizer: Any, model: Any) -> None:
    expected = {"batting": 1.0, "pitching": 2.0, "fielding": 3.0}
    for name, parameter in model.named_parameters():
        value = expected[name.split(".")[1]]
        state = optimizer.state[parameter]
        torch.testing.assert_close(state["exp_avg"], torch.full_like(parameter, value * 0.1))
        torch.testing.assert_close(
            state["exp_avg_sq"], torch.full_like(parameter, value**2 * 0.001),
        )


def test_optimizer_helper_imports_without_torch() -> None:
    code = textwrap.dedent("""
        import importlib.abc
        import sys
        class BlockTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise ImportError("optional torch deliberately unavailable")
        sys.meta_path.insert(0, BlockTorch())
        from cpv26.training.optimizer_state import make_adamw, optimizer_parameter_names
        assert callable(make_adamw) and callable(optimizer_parameter_names)
    """)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], env=environment, capture_output=True, text=True,
        check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("named", [True, False])
def test_restore_uses_saved_names_not_current_registration_order(torch: Any, named: bool) -> None:
    _, checkpoint = _checkpoint(torch, named=named)
    model = _model(torch, ("fielding", "batting", "pitching"))
    model.load_state_dict(checkpoint["model"])
    optimizer = make_adamw(
        model, learning_rate=0.01, weight_decay=0.02, checkpoint=checkpoint,
    )
    _assert_named_moments(torch, optimizer, model)
    assert optimizer_parameter_names(model, optimizer) == [[
        "roles.batting.weight", "roles.pitching.weight", "roles.fielding.weight",
    ]]


def test_legacy_reloaded_then_resaved_metadata_preserves_actual_optimizer_order(torch: Any) -> None:
    _, checkpoint = _checkpoint(torch, named=False)
    middle = _model(torch, ("fielding", "batting", "pitching"))
    middle.load_state_dict(checkpoint["model"])
    optimizer = make_adamw(
        middle, learning_rate=0.01, weight_decay=0.02, checkpoint=checkpoint,
    )
    resaved = {
        "model": middle.state_dict(), "optimizer": optimizer.state_dict(),
        "optimizer_parameter_names": optimizer_parameter_names(middle, optimizer),
    }
    assert list(dict(middle.named_parameters())) != resaved["optimizer_parameter_names"][0]
    third = _model(torch, ("pitching", "fielding", "batting"))
    third.load_state_dict(resaved["model"])
    restored = make_adamw(third, learning_rate=0.01, weight_decay=0.02, checkpoint=resaved)
    _assert_named_moments(torch, restored, third)


@pytest.mark.parametrize("metadata", [
    None, [], [["roles.batting.weight"]],
    [["roles.batting.weight", "roles.batting.weight", "roles.fielding.weight"]],
    [["roles.batting.weight", "unknown.weight", "roles.fielding.weight"]],
    [["roles.batting.weight", 42, "roles.fielding.weight"]],
    [["roles.batting.weight"], ["roles.pitching.weight", "roles.fielding.weight"]],
])
def test_invalid_name_metadata_fails_closed(torch: Any, metadata: Any) -> None:
    model, checkpoint = _checkpoint(torch)
    checkpoint["optimizer_parameter_names"] = metadata
    with pytest.raises(ValueError):
        make_adamw(model, learning_rate=0.01, weight_decay=0.02, checkpoint=checkpoint)


@pytest.mark.parametrize("damage", [
    "multiple_groups", "duplicate_id", "missing_id", "unknown_state_id",
    "bad_moment_shape", "legacy_missing_name", "legacy_unknown_name", "legacy_missing_model",
])
def test_invalid_optimizer_or_legacy_mapping_fails_closed(torch: Any, damage: str) -> None:
    model, checkpoint = _checkpoint(torch, named=not damage.startswith("legacy"))
    optimizer = checkpoint["optimizer"]
    if damage == "multiple_groups":
        optimizer["param_groups"].append(copy.deepcopy(optimizer["param_groups"][0]))
    elif damage == "duplicate_id":
        optimizer["param_groups"][0]["params"][1] = optimizer["param_groups"][0]["params"][0]
    elif damage == "missing_id":
        optimizer["param_groups"][0]["params"].pop()
    elif damage == "unknown_state_id":
        optimizer["state"][999] = {}
    elif damage == "bad_moment_shape":
        optimizer["state"][0]["exp_avg"] = torch.ones(5)
    elif damage == "legacy_missing_name":
        del checkpoint["model"]["roles.batting.weight"]
    elif damage == "legacy_unknown_name":
        checkpoint["model"]["unknown.weight"] = torch.ones(2, 2)
    else:
        del checkpoint["model"]
    with pytest.raises(ValueError):
        make_adamw(model, learning_rate=0.01, weight_decay=0.02, checkpoint=checkpoint)


def test_uninitialized_parameter_states_are_allowed(torch: Any) -> None:
    model = _model(torch)
    optimizer = make_adamw(model, learning_rate=0.01, weight_decay=0.02)
    checkpoint = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "optimizer_parameter_names": optimizer_parameter_names(model, optimizer),
    }
    assert checkpoint["optimizer"]["state"] == {}
    restored = make_adamw(model, learning_rate=0.01, weight_decay=0.02, checkpoint=checkpoint)
    assert len(restored.state) == 0


def test_clone_state_isolates_all_tensors_and_optimizer_updates(torch: Any) -> None:
    _, checkpoint = _checkpoint(torch)
    checkpoint["optimizer"]["state"][0]["extra"] = {"nested": [torch.tensor([7.0])]}
    original = copy.deepcopy(checkpoint)
    model = _model(torch, ("fielding", "pitching", "batting"))
    model.load_state_dict(checkpoint["model"])
    optimizer = make_adamw(
        model, learning_rate=0.01, weight_decay=0.02, checkpoint=checkpoint, clone_state=True,
    )
    first = optimizer.param_groups[0]["params"][0]
    assert optimizer.state[first]["step"].data_ptr() != checkpoint["optimizer"]["state"][0][
        "step"
    ].data_ptr()
    optimizer.state[first]["extra"]["nested"][0].fill_(-1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for identifier, state in original["optimizer"]["state"].items():
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(checkpoint["optimizer"]["state"][identifier][key], state[key])
    assert torch.equal(
        checkpoint["optimizer"]["state"][0]["extra"]["nested"][0], torch.tensor([7.0]),
    )
    for name, tensor in original["model"].items():
        assert torch.equal(checkpoint["model"][name], tensor)


def test_optimizer_names_reject_partial_foreign_and_multiple_groups(torch: Any) -> None:
    model = _model(torch)
    parameters = list(model.parameters())
    invalid = [
        torch.optim.AdamW(parameters[:1]),
        torch.optim.AdamW([*parameters, torch.nn.Parameter(torch.ones(2, 2))]),
        torch.optim.AdamW([{"params": parameters[:1]}, {"params": parameters[1:]}]),
    ]
    for optimizer in invalid:
        with pytest.raises(ValueError):
            optimizer_parameter_names(model, optimizer)


def test_legacy_inference_follows_stored_model_order_with_buffers_interspersed(torch: Any) -> None:
    _, checkpoint = _checkpoint(torch, named=False)
    assert isinstance(checkpoint["model"], OrderedDict)
    model = _model(torch, ("pitching", "fielding", "batting"))
    restored = make_adamw(model, learning_rate=0.01, weight_decay=0.02, checkpoint=checkpoint)
    names = optimizer_parameter_names(model, restored)[0]
    assert not any("buffer" in name for name in names)
    assert names == ["roles.batting.weight", "roles.pitching.weight", "roles.fielding.weight"]
    _assert_named_moments(torch, restored, model)
