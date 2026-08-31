from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any

import pytest

from cpv26.graph import AtomicRoute, RouteRegistry, TorchAtomicRouteBatch


@pytest.fixture(scope="module")
def torch_runtime() -> Iterator[Any]:
    torch = pytest.importorskip("torch")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield torch
    finally:
        torch.set_num_threads(previous_threads)


def _backbone(
    *,
    event_dim: int = 3,
    layers: int = 2,
    bidirectional: bool = True,
    dropout: float = 0.25,
    publication_delay: bool = True,
) -> Any:
    from cpv26.models import CompositeRelGNNBackbone

    registry = RouteRegistry(
        (
            AtomicRoute(
                "player_interaction", "player", "event", "player", "batting", "pitching",
                bidirectional=bidirectional,
            ),
            AtomicRoute(
                "team_interaction", "team", "event", "team", "home", "away",
                bidirectional=bidirectional,
            ),
            AtomicRoute(
                "empty_candidate", "player", "event", "team", "batting", "team",
                bidirectional=bidirectional,
            ),
        )
    )
    return CompositeRelGNNBackbone(
        node_feature_dims={"player": 3, "team": 2},
        route_feature_dims={name: event_dim for name in registry.names()},
        hidden_dim=8,
        num_layers=layers,
        num_attention_heads=2,
        dropout=dropout,
        include_publication_delay=publication_delay,
        registry=registry,
    )


def _inputs(
    torch: Any, model: Any, *, empty: bool = False, device: str = "cpu"
) -> tuple[dict[str, Any], dict[str, Any], tuple[TorchAtomicRouteBatch, ...], dict[str, Any]]:
    leaves: dict[str, Any] = {}

    def leaf(name: str, value: Any) -> Any:
        tensor = value.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
        leaves[name] = tensor
        return tensor

    nodes = {
        "player": leaf("node.player", torch.randn(4, 3)),
        "team": leaf("node.team", torch.randn(3, 2)),
    }
    roles = {role: leaf(f"role.{role}", torch.randn(4, 8)) for role in ("batting", "pitching")}
    batches = []
    for route in model.registry:
        is_empty = empty or route.name == "empty_candidate"
        count = 0 if is_empty else 6
        if route.source_type == "player":
            source, destination = [0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 1, 3]
        else:
            source, destination = [0, 1, 2, 0, 1, 2], [1, 2, 0, 2, 0, 1]
        batches.append(
            TorchAtomicRouteBatch(
                route_name=route.name,
                source_type=route.source_type,
                destination_type=route.destination_type,
                source_index=torch.tensor(source[:count], dtype=torch.long, device=device),
                destination_index=torch.tensor(
                    destination[:count], dtype=torch.long, device=device
                ),
                event_features=leaf(
                    f"{route.name}.event",
                    torch.randn(count, model.route_feature_dims[route.name]),
                ),
                event_age_seconds=leaf(
                    f"{route.name}.age", torch.linspace(7200.0, 400000.0, count)
                ),
                publication_delay_seconds=leaf(
                    f"{route.name}.delay", torch.linspace(120.0, 2400.0, count)
                ),
                # Include duplicate endpoints, a zero weight, and non-unit weights.
                weights=leaf(
                    f"{route.name}.weights", torch.tensor([1.0, 0.5, 0.0, 2.0, 0.75, 0.25][:count])
                ),
                bidirectional=route.bidirectional,
            )
        )
    return nodes, roles, tuple(batches), leaves


def _forward_backward(
    torch: Any, model: Any, inputs: Any, *, seed: int, device: str = "cpu", amp: bool = False
) -> tuple[dict[str, Any], Any, list[Any]]:
    nodes, roles, batches, leaves = inputs
    model.zero_grad(set_to_none=True)
    for tensor in leaves.values():
        tensor.grad = None
    torch.manual_seed(seed)
    context = torch.autocast(device, dtype=torch.bfloat16) if amp else nullcontext()
    with context:
        state = model.forward_relational_state(nodes, batches, player_role_states=roles)
        outputs = state.channels()
        terms = []
        for value in outputs.values():
            target = torch.linspace(-0.7, 1.3, value.numel(), device=device).reshape(value.shape)
            terms.append((value.float() * target).mean() + 0.1 * value.float().square().mean())
        loss = sum(terms)
    loss.backward()
    rng = [torch.get_rng_state().clone()]
    if device == "cuda":
        rng.append(torch.cuda.get_rng_state().clone())
    return outputs, loss.detach(), rng


def _assert_gradients_close(
    torch: Any, original: dict[str, Any], optimized: dict[str, Any], *, amp: bool = False
) -> None:
    assert original.keys() == optimized.keys()
    reference_gradients, optimized_gradients = [], []
    for name, value in original.items():
        actual = optimized[name]
        assert (value.grad is None) == (actual.grad is None), name
        if value.grad is None:
            continue
        assert torch.isfinite(value.grad).all(), name
        assert torch.isfinite(actual.grad).all(), name
        if amp:
            reference_gradients.append(value.grad.float().flatten())
            optimized_gradients.append(actual.grad.float().flatten())
        else:
            torch.testing.assert_close(value.grad, actual.grad, rtol=3e-5, atol=3e-6, msg=name)
    if amp:
        reference = torch.cat(reference_gradients)
        actual = torch.cat(optimized_gradients)
        # Reusing a BF16 intermediate sums the two downstream gradients before
        # its backward, rather than after two separate backwards. This changes
        # rounding, not the mathematical derivative; do not require bit equality.
        error = torch.linalg.vector_norm(reference - actual)
        bound = 0.02 * torch.linalg.vector_norm(reference) + 1e-6
        assert error <= bound, (float(error), float(bound))


@pytest.mark.parametrize(
    ("layers", "bidirectional", "event_dim", "empty", "dropout", "publication_delay"),
    (
        (2, True, 3, False, 0.25, True),
        (2, True, 0, False, 0.25, False),
        (2, False, 3, False, 0.25, True),
        (2, True, 3, True, 0.25, True),
        (2, True, 3, False, 0.0, False),
        (1, True, 3, False, 0.25, True),
    ),
)
def test_reuse_matches_original_outputs_gradients_and_adamw_steps(
    torch_runtime: Any,
    layers: int,
    bidirectional: bool,
    event_dim: int,
    empty: bool,
    dropout: float,
    publication_delay: bool,
) -> None:
    torch = torch_runtime
    torch.manual_seed(314)
    optimized = _backbone(
        layers=layers, bidirectional=bidirectional, event_dim=event_dim,
        dropout=dropout, publication_delay=publication_delay,
    )
    original = copy.deepcopy(optimized)
    original.set_execution_optimization(False)
    optimized_inputs = _inputs(torch, optimized, empty=empty)
    original_inputs = copy.deepcopy(optimized_inputs)
    original_optimizer = torch.optim.AdamW(original.parameters(), lr=1e-3, weight_decay=0.01)
    optimized_optimizer = torch.optim.AdamW(optimized.parameters(), lr=1e-3, weight_decay=0.01)

    # A second step also detects stale cached tensors/autograd graphs.
    for step in range(2):
        expected, expected_loss, expected_rng = _forward_backward(
            torch, original, original_inputs, seed=2010 + step
        )
        actual, actual_loss, actual_rng = _forward_backward(
            torch, optimized, optimized_inputs, seed=2010 + step
        )
        for channel in expected:
            torch.testing.assert_close(expected[channel], actual[channel], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(expected_loss, actual_loss, rtol=1e-5, atol=2e-6)
        assert all(
            torch.equal(left, right)
            for left, right in zip(expected_rng, actual_rng, strict=True)
        )
        _assert_gradients_close(torch, original_inputs[-1], optimized_inputs[-1])
        _assert_gradients_close(
            torch, dict(original.named_parameters()), dict(optimized.named_parameters())
        )
        original_optimizer.step()
        optimized_optimizer.step()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(
                value, optimized.state_dict()[name], rtol=3e-5, atol=2e-5, msg=name
            )
        original_states = original_optimizer.state_dict()["state"]
        optimized_states = optimized_optimizer.state_dict()["state"]
        assert original_states.keys() == optimized_states.keys()
        for index, state in original_states.items():
            for key, value in state.items():
                torch.testing.assert_close(
                    value, optimized_states[index][key], rtol=3e-5, atol=3e-6
                )


@pytest.mark.parametrize("bidirectional", (True, False))
def test_context_encoders_run_once_per_nonempty_route_per_layer_and_forward(
    torch_runtime: Any, bidirectional: bool
) -> None:
    torch = torch_runtime
    model = _backbone(bidirectional=bidirectional)
    nodes, roles, batches, _ = _inputs(torch, model)
    counts: dict[str, int] = {}
    dropout_order: list[str] = []
    handles = []

    def count_hook(name: str) -> Any:
        def hook(_module: Any, _arguments: Any, _output: Any) -> None:
            counts[name] += 1
        return hook

    def dropout_hook(name: str) -> Any:
        def hook(_module: Any, _arguments: Any, _output: Any) -> None:
            dropout_order.append(name)
        return hook

    for layer_index, layer in enumerate(model.layers):
        for route_name, attention in layer.messages.items():
            for encoder_name in ("event_encoder", "temporal_encoder"):
                name = f"{layer_index}.{route_name}.{encoder_name}"
                counts[name] = 0
                handles.append(
                    getattr(attention, encoder_name).register_forward_hook(count_hook(name))
                )
            handles.append(
                attention.dropout.register_forward_hook(dropout_hook(f"{layer_index}.{route_name}"))
            )
    observed_orders = []
    try:
        for enabled in (False, True):
            model.set_execution_optimization(enabled)
            counts = dict.fromkeys(counts, 0)
            dropout_order.clear()
            for _ in range(2):
                model.forward_relational_state(nodes, batches, player_role_states=roles)
            for name, count in counts.items():
                expected = 0 if "empty_candidate" in name else 2
                if bidirectional and not enabled:
                    expected *= 2
                assert count == expected, name
            observed_orders.append(list(dropout_order))
        assert observed_orders[0] == observed_orders[1]
        assert len(observed_orders[0]) == 2 * 2 * 2 * (2 if bidirectional else 1)
    finally:
        for handle in handles:
            handle.remove()


def test_optimization_toggle_does_not_change_checkpoint_keys_or_parameters(
    torch_runtime: Any,
) -> None:
    torch = torch_runtime
    model = _backbone()
    assert model.execution_optimization_enabled is True
    original_state = copy.deepcopy(model.state_dict())
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    model.set_execution_optimization(False)
    assert model.execution_optimization_enabled is False
    assert original_state.keys() == model.state_dict().keys()
    assert parameter_ids == {name: id(value) for name, value in model.named_parameters()}
    model.load_state_dict(original_state, strict=True)
    assert model.execution_optimization_enabled is False
    restored = _backbone()
    restored.load_state_dict(original_state, strict=True)
    assert restored.execution_optimization_enabled is True
    for name, value in original_state.items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(TypeError, match="bool"):
        model.set_execution_optimization("false")


@pytest.mark.parametrize(("device", "amp"), (("cpu", True), ("cuda", False), ("cuda", True)))
def test_autocast_and_optional_cuda_reuse_preserve_outputs_rng_and_finite_gradients(
    torch_runtime: Any, device: str, amp: bool
) -> None:
    torch = torch_runtime
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA device is unavailable")
        if amp and not torch.cuda.is_bf16_supported():
            pytest.skip("CUDA device does not support BF16")
    torch.manual_seed(831)
    optimized = _backbone().to(device)
    original = copy.deepcopy(optimized)
    original.set_execution_optimization(False)
    optimized_inputs = _inputs(torch, optimized, device=device)
    original_inputs = copy.deepcopy(optimized_inputs)
    expected, expected_loss, expected_rng = _forward_backward(
        torch, original, original_inputs, seed=57, device=device, amp=amp
    )
    actual, actual_loss, actual_rng = _forward_backward(
        torch, optimized, optimized_inputs, seed=57, device=device, amp=amp
    )
    for channel in expected:
        assert torch.isfinite(actual[channel]).all()
        torch.testing.assert_close(expected[channel], actual[channel], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(expected_loss, actual_loss, rtol=1e-5, atol=2e-6)
    assert all(
        torch.equal(left, right)
        for left, right in zip(expected_rng, actual_rng, strict=True)
    )
    _assert_gradients_close(torch, original_inputs[-1], optimized_inputs[-1], amp=amp)
    _assert_gradients_close(
        torch, dict(original.named_parameters()), dict(optimized.named_parameters()), amp=amp
    )
