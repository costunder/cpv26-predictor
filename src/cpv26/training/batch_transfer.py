"""Optional, dtype-preserving packed transfers of read-only CPU batch tensors.

Packing changes transport only. Each eligible dtype uses one private pinned CPU
allocation and one asynchronous CUDA transfer. Reconstructed tensors occupy
non-overlapping views, except repeated references to the very same input tensor.
Unsupported tensors retain the ordinary ``Tensor.to`` path and autograd behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

from cpv26.models._torch import require_torch


@dataclass(frozen=True, slots=True)
class _TensorSlice:
    source_id: int
    shape: tuple[int, ...]
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class _PackedGroup:
    buffer: Any
    slices: tuple[_TensorSlice, ...]


def _map_tensors(value: Any, operation: Callable[[Any], Any], torch: Any) -> Any:
    """Use the same supported container semantics as the original runner move."""
    if torch.is_tensor(value):
        return operation(value)
    if isinstance(value, Mapping):
        return {key: _map_tensors(item, operation, torch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, operation, torch) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, operation, torch) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{
            field.name: _map_tensors(getattr(value, field.name), operation, torch)
            for field in fields(value)
        })
    return value


def _tensor_leaves(value: Any, torch: Any) -> Iterator[Any]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _tensor_leaves(item, torch)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item, torch)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _tensor_leaves(getattr(value, field.name), torch)


def _can_pack(tensor: Any, torch: Any) -> bool:
    # Avoid erasing subclass, sparse, named, conjugate or gradient semantics.
    return (
        type(tensor) is torch.Tensor
        and tensor.device.type == "cpu"
        and tensor.layout == torch.strided
        and not tensor.is_nested
        and not tensor.requires_grad
        and tensor.dtype in {
            torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float16, torch.bfloat16, torch.float32, torch.float64,
        }
        and tensor.is_contiguous()
        and tensor.numel() > 0
        and not tensor.is_conj()
        and not tensor.is_neg()
        and all(name is None for name in getattr(tensor, "names", ()))
    )


def _pack_cpu_tensors(tensors: Iterable[Any], *, pin_memory: bool) -> list[_PackedGroup]:
    """Separate CPU-testable packing from CUDA; never mutate a source tensor."""
    torch, _ = require_torch()
    by_dtype: dict[Any, list[Any]] = {}
    seen: set[int] = set()
    for tensor in tensors:
        identity = id(tensor)
        if identity not in seen and _can_pack(tensor, torch):
            seen.add(identity)
            by_dtype.setdefault(tensor.dtype, []).append(tensor)
    groups: list[_PackedGroup] = []
    for dtype, members in by_dtype.items():
        slices: list[_TensorSlice] = []
        offset = 0
        for tensor in members:
            length = tensor.numel()
            slices.append(_TensorSlice(id(tensor), tuple(tensor.shape), offset, length))
            offset += length
        buffer = torch.empty(offset, dtype=dtype, device="cpu", pin_memory=pin_memory)
        torch.cat([tensor.view(-1) for tensor in members], out=buffer)
        groups.append(_PackedGroup(buffer, tuple(slices)))
    return groups


def _unpack_groups(groups: Iterable[_PackedGroup], device: Any) -> dict[int, Any]:
    restored: dict[int, Any] = {}
    for group in groups:
        # PyTorch's own pinned allocator records the non-blocking copy's stream
        # event; releasing this private, never-mutated buffer cannot recycle its
        # memory before DMA completes. CUDA consumers use the current stream,
        # just as with ordinary Tensor.to(non_blocking=True).
        moved = group.buffer.to(device=device, non_blocking=True).detach()
        for view in group.slices:
            restored[view.source_id] = moved.narrow(0, view.offset, view.length).view(view.shape)
    return restored


def move_batch(value: Any, device: Any, *, packed: bool = True) -> Any:
    """Move the runner's nested batch, optionally coalescing CPU-to-CUDA copies.

    ``packed=False`` preserves the original recursive transport. CPU targets and
    unsupported tensor representations use that same path. Packed tensors retain
    dtype, shape and values, but share private storage as non-overlapping views;
    batch inputs are read-only. No buffer pool or global cache is retained.
    """
    torch, _ = require_torch()
    selected = torch.device(device)
    if not packed or selected.type != "cuda":
        return _map_tensors(
            value, lambda tensor: tensor.to(device=device, non_blocking=True), torch,
        )
    groups = _pack_cpu_tensors(_tensor_leaves(value, torch), pin_memory=True)
    moved = _unpack_groups(groups, selected)

    def lookup(tensor: Any) -> Any:
        identity = id(tensor)
        if identity not in moved:
            moved[identity] = tensor.to(device=selected, non_blocking=True)
        return moved[identity]

    return _map_tensors(value, lookup, torch)


def _record_cuda_batch_stream(value: Any, stream: Any, torch: Any) -> None:
    """Tie every CUDA storage in a nested batch to its consumer stream."""

    seen: set[int] = set()
    for tensor in _tensor_leaves(value, torch):
        identity = id(tensor)
        if identity in seen or tensor.device.type != "cuda":
            continue
        seen.add(identity)
        tensor.record_stream(stream)


def prefetch_batches(
    batches: Iterable[Any],
    device: Any,
    *,
    mover: Callable[[Any, Any], Any] | None = None,
) -> Iterator[Any]:
    """Move one CUDA batch ahead on a dedicated copy stream.

    The iterator preserves source order and yields the exact structure returned
    by ``mover``.  On CUDA it holds at most the current and next device batches:
    batch N+1's H2D copies may overlap batch N's compute, while an explicit
    stream dependency prevents any consumer from observing an incomplete copy.
    ``record_stream`` protects packed shared storages from allocator reuse until
    the compute stream is finished with them.  CPU execution remains a simple,
    non-look-ahead mapping so validation semantics and object lifetimes stay
    unchanged.
    """

    torch, _ = require_torch()
    selected = torch.device(device)
    move = mover or (lambda value, target: move_batch(value, target, packed=True))
    iterator = iter(batches)
    if selected.type != "cuda":
        for raw_batch in iterator:
            moved_batch = move(raw_batch, selected)
            yield moved_batch
            del moved_batch
        return

    copy_stream = torch.cuda.Stream(device=selected)
    missing = object()
    next_batch: Any = missing

    def preload() -> None:
        nonlocal next_batch
        try:
            raw_batch = next(iterator)
        except StopIteration:
            next_batch = missing
            return
        with torch.cuda.stream(copy_stream):
            next_batch = move(raw_batch, selected)

    preload()
    while next_batch is not missing:
        compute_stream = torch.cuda.current_stream(selected)
        compute_stream.wait_stream(copy_stream)
        current_batch = next_batch
        _record_cuda_batch_stream(current_batch, compute_stream, torch)
        # Queue the next copy before yielding.  The caller subsequently enqueues
        # current-batch compute on ``compute_stream``, allowing the streams to
        # overlap without changing batch or optimizer order.
        preload()
        yield current_batch
        del current_batch
