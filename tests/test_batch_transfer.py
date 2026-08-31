from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from cpv26.graph.snapshot import TorchAtomicRouteBatch
from cpv26.training import batch_transfer
from cpv26.training.batch_transfer import _pack_cpu_tensors, _unpack_groups, move_batch


@pytest.fixture
def torch() -> Any:
    return pytest.importorskip("torch")


def test_transfer_module_imports_without_optional_torch() -> None:
    code = textwrap.dedent("""
        import importlib.abc
        import sys

        class BlockTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise ImportError("optional torch deliberately unavailable")

        sys.meta_path.insert(0, BlockTorch())
        from cpv26.training.batch_transfer import move_batch
        from cpv26.models._torch import TorchUnavailableError
        try:
            move_batch({}, "cpu")
        except TorchUnavailableError:
            pass
        else:
            raise AssertionError("runtime use must fail clearly without torch")
    """)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], env=environment, capture_output=True, text=True,
        check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("dtype_name", [
    "bool", "uint8", "int8", "int16", "int32", "int64",
    "float16", "bfloat16", "float32", "float64",
])
def test_cpu_pack_preserves_dtype_values_shapes_and_scalar(torch: Any, dtype_name: str) -> None:
    dtype = getattr(torch, dtype_name)
    matrix = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    scalar = torch.tensor(1, dtype=dtype)
    originals = [matrix.clone(), scalar.clone()]
    groups = _pack_cpu_tensors([matrix, scalar], pin_memory=False)
    restored = _unpack_groups(groups, "cpu")
    assert len(groups) == 1
    assert groups[0].buffer.dtype == dtype
    assert groups[0].buffer.numel() == 5
    for source, snapshot in zip((matrix, scalar), originals, strict=True):
        actual = restored[id(source)]
        assert actual.dtype == source.dtype and actual.shape == source.shape
        assert torch.equal(actual, snapshot)
        assert torch.equal(source, snapshot)
        assert not actual.requires_grad


def test_pack_groups_are_per_dtype_and_deduplicate_same_object(torch: Any) -> None:
    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([3.0])
    integer = torch.tensor([7, 8], dtype=torch.int64)
    mask = torch.tensor([True, False])
    groups = _pack_cpu_tensors([first, second, first, integer, mask], pin_memory=False)
    assert len(groups) == 3
    float_group = next(group for group in groups if group.buffer.dtype == torch.float32)
    assert float_group.buffer.numel() == 3
    assert len(float_group.slices) == 2
    restored = _unpack_groups(groups, "cpu")
    assert len(restored) == 4
    # Different tensors share transport storage, never overlapping value ranges.
    restored[id(first)].fill_(9)
    assert torch.equal(restored[id(second)], second)
    assert torch.equal(first, torch.tensor([1.0, 2.0]))
    assert torch.equal(integer, torch.tensor([7, 8]))


def test_pack_does_not_reuse_source_storage_or_subsequent_batch_storage(torch: Any) -> None:
    first = torch.arange(6.0).reshape(2, 3)
    second = torch.arange(3.0)
    one = _unpack_groups(_pack_cpu_tensors([first, second], pin_memory=False), "cpu")
    two = _unpack_groups(_pack_cpu_tensors([first, second], pin_memory=False), "cpu")
    first.fill_(-1)
    assert torch.equal(one[id(first)], torch.arange(6.0).reshape(2, 3))
    one[id(first)].fill_(-2)
    assert torch.equal(two[id(first)], torch.arange(6.0).reshape(2, 3))


def test_unsupported_tensor_representations_are_not_packed(torch: Any) -> None:
    eligible = torch.tensor([1.0, 2.0])
    unsupported = [
        torch.empty(0),
        torch.arange(6.0).reshape(2, 3).t(),
        torch.ones(2, requires_grad=True),
        torch.tensor([1 + 2j]),
        torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (2,)),
        torch.nn.Parameter(torch.ones(2), requires_grad=False),
        torch.empty(2, device="meta"),
    ]
    groups = _pack_cpu_tensors([eligible, *unsupported], pin_memory=False)
    restored = _unpack_groups(groups, "cpu")
    assert set(restored) == {id(eligible)}
    assert all(id(tensor) not in restored for tensor in unsupported)


@dataclass(frozen=True, slots=True)
class _MetadataBatch:
    tensor: Any
    label: str
    metadata: Any


@pytest.mark.parametrize("packed", [False, True])
def test_cpu_move_preserves_container_metadata_and_autograd(
    torch: Any, monkeypatch: pytest.MonkeyPatch, packed: bool,
) -> None:
    def no_packing(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("CPU destination must use the ordinary recursive path")

    monkeypatch.setattr(batch_transfer, "_pack_cpu_tensors", no_packing)
    gradient = torch.tensor([2.0], requires_grad=True)
    marker = object()
    value = {"items": [_MetadataBatch(gradient, "original", marker), (gradient, None)], "seed": 7}
    actual = move_batch(value, "cpu", packed=packed)
    assert isinstance(actual, dict) and isinstance(actual["items"], list)
    assert isinstance(actual["items"][0], _MetadataBatch)
    assert isinstance(actual["items"][1], tuple)
    assert actual["items"][0].tensor is gradient
    assert actual["items"][0].metadata is marker
    assert actual["items"][0].label == "original" and actual["seed"] == 7
    actual["items"][0].tensor.sum().backward()
    assert torch.equal(gradient.grad, torch.ones_like(gradient))


def _route(torch: Any) -> TorchAtomicRouteBatch:
    source = torch.tensor([0, 1], dtype=torch.int64)
    return TorchAtomicRouteBatch(
        "real-route", "player", "team", source, source,
        torch.tensor([[1.0], [2.0]]), torch.tensor([3.0, 4.0]),
        torch.zeros(2), torch.ones(2), False,
    )


def test_cpu_route_dataclass_is_preserved(torch: Any) -> None:
    route = _route(torch)
    result = move_batch({"routes": [route], "days": ("2024-07-01",)}, "cpu")
    actual = result["routes"][0]
    assert isinstance(actual, TorchAtomicRouteBatch)
    assert actual.route_name == route.route_name
    assert actual.source_type == "player" and actual.destination_type == "team"
    assert not actual.bidirectional
    assert torch.equal(actual.source_index, route.source_index)
    assert result["days"] == ("2024-07-01",)


def test_cpu_packed_reconstruction_preserves_metadata_and_repeated_tensor_alias(torch: Any) -> None:
    route = _route(torch)
    marker = object()
    source = {"route": route, "metadata": ("2024-07-01", marker), "items": [route.weights]}
    groups = _pack_cpu_tensors(batch_transfer._tensor_leaves(source, torch), pin_memory=False)
    tensors = _unpack_groups(groups, "cpu")
    result = batch_transfer._map_tensors(source, lambda tensor: tensors[id(tensor)], torch)
    actual = result["route"]
    assert isinstance(actual, TorchAtomicRouteBatch)
    assert actual.source_index is actual.destination_index
    assert actual.weights is result["items"][0]
    assert result["metadata"] == ("2024-07-01", marker)
    assert actual.route_name == route.route_name and actual.bidirectional is False
    assert torch.equal(actual.event_features, route.event_features)
    assert actual.event_features.data_ptr() != route.event_features.data_ptr()


def test_cuda_packed_matches_legacy_and_preserves_alias_and_input(torch: Any) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is unavailable")
    route = _route(torch)
    value = {"route": route, "mask": torch.tensor([True, False]), "label": "unchanged"}
    packed = move_batch(value, "cuda:0", packed=True)
    legacy = move_batch(value, "cuda:0", packed=False)
    torch.cuda.synchronize("cuda:0")
    for name in (
        "source_index", "destination_index", "event_features", "event_age_seconds",
        "publication_delay_seconds", "weights",
    ):
        actual, expected = getattr(packed["route"], name), getattr(legacy["route"], name)
        assert actual.device.type == "cuda" and actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
        assert getattr(route, name).device.type == "cpu"
    assert packed["route"].source_index is packed["route"].destination_index
    assert torch.equal(packed["mask"], legacy["mask"])
    assert packed["label"] == "unchanged"


def test_cuda_pinned_staging_survives_async_copy_and_source_mutation(torch: Any) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is unavailable")
    stream = torch.cuda.Stream(device="cuda:0")
    first = torch.arange(131072, dtype=torch.float32)
    second = first + 1
    expected_first, expected_second = first.clone(), second.clone()
    groups = _pack_cpu_tensors([first, second], pin_memory=True)
    assert all(group.buffer.is_pinned() for group in groups)
    with torch.cuda.stream(stream):
        # Delay this stream, so host-side cleanup happens while DMA is queued.
        torch.cuda._sleep(100_000_000)
        restored = _unpack_groups(groups, "cuda:0")
    first.zero_()
    second.zero_()
    del groups
    gc.collect()
    # Repeated allocator requests must not recycle a pending transfer's source.
    churn = [torch.full((262144,), -5.0, pin_memory=True) for _ in range(8)]
    stream.synchronize()
    assert torch.equal(restored[id(first)].cpu(), expected_first)
    assert torch.equal(restored[id(second)].cpu(), expected_second)
    assert all(tensor.is_pinned() for tensor in churn)


def test_cuda_fallback_preserves_noncontiguous_values_and_gradient(torch: Any) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is unavailable")
    gradient = torch.arange(6.0, requires_grad=True)
    noncontiguous = torch.arange(6.0).reshape(2, 3).t()
    empty = torch.empty(0)
    actual = move_batch([gradient, noncontiguous, empty], "cuda:0")
    assert actual[0].requires_grad and actual[2].numel() == 0
    assert torch.equal(actual[1].cpu(), noncontiguous)
    actual[0].sum().backward()
    assert torch.equal(gradient.grad, torch.ones_like(gradient))
