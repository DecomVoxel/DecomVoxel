"""Threading helpers for DecomVoxel parallel pipeline.

Provides a small wrapper around ``concurrent.futures.ThreadPoolExecutor`` so
that the per-object loops in the demo pipeline can be parallelised with a
single configurable ``max_workers`` value, while keeping per-task error
handling explicit and ordered.
"""

from __future__ import annotations

import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def run_parallel(
    fn: Callable[..., Any],
    items: Iterable[Tuple[Any, ...]],
    max_workers: int = 4,
    desc: str = "task",
) -> List[Tuple[Any, Any, Exception]]:
    """Run ``fn(*args)`` over ``items`` in a thread pool.

    Parameters
    ----------
    fn : callable
        The function to invoke. Each element of ``items`` must be a tuple of
        positional arguments forwarded to ``fn``.
    items : iterable of tuple
        Iterable of argument tuples. The first element of each tuple is used
        as a human-readable key in log messages and returned alongside the
        result.
    max_workers : int
        Maximum number of concurrent worker threads. Values < 1 are clamped
        to 1 (sequential).
    desc : str
        Short label used in log lines.

    Returns
    -------
    list of (key, result, exception)
        One entry per submitted item, in submission order. ``result`` is
        ``None`` when the task raised; ``exception`` is ``None`` when the
        task succeeded.
    """
    items_list = list(items)
    max_workers = max(1, int(max_workers))
    print(f"[parallel/{desc}] launching {len(items_list)} tasks with max_workers={max_workers}")

    results: List[Tuple[Any, Any, Exception]] = [None] * len(items_list)  # type: ignore[list-item]

    def _wrap(idx: int, args: Tuple[Any, ...]):
        try:
            return idx, args[0], fn(*args), None
        except Exception as exc:  # pragma: no cover - logging path
            print(f"[parallel/{desc}] task '{args[0]}' raised: {exc}")
            traceback.print_exc()
            return idx, args[0], None, exc

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_wrap, i, args) for i, args in enumerate(items_list)]
        for fut in as_completed(futures):
            idx, key, value, exc = fut.result()
            results[idx] = (key, value, exc)
            status = "OK" if exc is None else "FAIL"
            print(f"[parallel/{desc}] [{idx + 1}/{len(items_list)}] {key}: {status}")

    return results


class OrderedPrinter:
    """Simple lock to serialise multi-line stdout from worker threads."""

    def __init__(self):
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._lock:
            print(*args, **kwargs)


def get_visible_devices() -> List[str]:
    """Return a list of torch device strings for every visible CUDA GPU.

    The list respects ``CUDA_VISIBLE_DEVICES`` (PyTorch already does this);
    the returned indices are PyTorch-local indices (``cuda:0``, ``cuda:1``,
    …). If CUDA is unavailable, returns ``["cpu"]``.
    """
    try:
        import torch  # local import: keep this module importable without torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if n > 0:
                return [f"cuda:{i}" for i in range(n)]
    except Exception:
        pass
    return ["cpu"]


def run_parallel_per_gpu(
    fn: Callable[..., Any],
    items: Iterable[Tuple[Any, ...]],
    parallel_num_per_gpu: int = 4,
    desc: str = "task",
    devices: Optional[List[str]] = None,
) -> List[Tuple[Any, Any, Optional[Exception]]]:
    """Run ``fn(device, *args)`` over ``items`` across every visible GPU.

    A token queue is filled with ``parallel_num_per_gpu`` copies of each
    device string. Each task acquires one token (blocking when all are
    busy), runs ``fn(device, *args)`` on the corresponding GPU, then
    returns the token to the queue. Total concurrency is
    ``len(devices) * parallel_num_per_gpu``.

    Parameters
    ----------
    fn : callable
        ``fn(device_str, *args) -> Any``. The first parameter receives the
        assigned device string (e.g. ``"cuda:1"``).
    items : iterable of tuple
        Each tuple is ``(key, *args)``; ``key`` is used as the log label.
    parallel_num_per_gpu : int
        Maximum concurrent tasks per GPU. Clamped to ≥ 1.
    desc : str
        Short label used in log lines.
    devices : list of str, optional
        Explicit device list. Defaults to :func:`get_visible_devices`.
    """
    items_list = list(items)
    parallel_num_per_gpu = max(1, int(parallel_num_per_gpu))
    devices = devices if devices is not None else get_visible_devices()

    slot_queue: "queue.Queue[str]" = queue.Queue()
    for d in devices:
        for _ in range(parallel_num_per_gpu):
            slot_queue.put(d)
    max_workers = len(devices) * parallel_num_per_gpu

    print(
        f"[parallel/{desc}] launching {len(items_list)} tasks across "
        f"{len(devices)} device(s) × {parallel_num_per_gpu} workers/device "
        f"(total max_workers={max_workers}) on devices={devices}"
    )

    results: List[Tuple[Any, Any, Optional[Exception]]] = [None] * len(items_list)  # type: ignore[list-item]

    def _wrap(idx: int, args: Tuple[Any, ...]):
        key = args[0]
        device = slot_queue.get()
        try:
            try:
                import torch
                if device.startswith("cuda"):
                    torch.cuda.set_device(device)
            except Exception:
                pass
            print(f"[parallel/{desc}] [{idx + 1}/{len(items_list)}] '{key}' START on {device}")
            value = fn(device, *args)
            return idx, key, value, None, device
        except Exception as exc:  # pragma: no cover - logging path
            print(f"[parallel/{desc}] task '{key}' on {device} raised: {exc}")
            traceback.print_exc()
            return idx, key, None, exc, device
        finally:
            slot_queue.put(device)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_wrap, i, args) for i, args in enumerate(items_list)]
        for fut in as_completed(futures):
            idx, key, value, exc, device = fut.result()
            results[idx] = (key, value, exc)
            status = "OK" if exc is None else "FAIL"
            print(f"[parallel/{desc}] [{idx + 1}/{len(items_list)}] {key} on {device}: {status}")

    return results


class PerDeviceCache:
    """Thread-safe per-device cache for heavy preprocessing artefacts.

    Typical use: build an object once on the main GPU, then lazily produce
    per-device copies for worker threads on other GPUs.

    Example::

        cache = PerDeviceCache(vis_grid, lambda obj, dev: obj.to(dev))
        local_vis = cache.get(device)
    """

    def __init__(self, base_obj: Any, mover: Callable[[Any, str], Any]):
        self._base = base_obj
        self._mover = mover
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, device: str) -> Any:
        device = str(device)
        with self._lock:
            cached = self._cache.get(device)
            if cached is not None:
                return cached
        moved = self._mover(self._base, device)
        with self._lock:
            # In case another thread raced and produced the same device copy,
            # keep the first-written one to avoid duplicate large allocations.
            existing = self._cache.get(device)
            if existing is not None:
                return existing
            self._cache[device] = moved
            return moved
