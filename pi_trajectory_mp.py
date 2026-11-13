# -*- coding: utf-8 -*-
"""Shared-memory multiprocessing for exact pi trajectory generation.

Pass 1 computes compact displacement summaries for independent digit blocks.
The parent composes each block's starting angle and coordinate. Pass 2 lets
workers calculate the final global coordinates directly into shared memory.
Large arrays are never serialized between processes.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter
from typing import Final

import numpy as np

_DIRECTION_ANGLES: Final[np.ndarray] = (
    np.arange(10, dtype=np.float64) * (2.0 * np.pi / 10.0)
)
_STEP_X: Final[np.ndarray] = np.cos(_DIRECTION_ANGLES)
_STEP_Y: Final[np.ndarray] = np.sin(_DIRECTION_ANGLES)

_WORKER_DIGITS_SHM: SharedMemory | None = None
_WORKER_POSITIONS_SHM: SharedMemory | None = None
_WORKER_DIGITS: np.ndarray | None = None
_WORKER_POSITIONS: np.ndarray | None = None
_PARENT_OUTPUT_HANDLES: list[SharedMemory] = []


def physical_core_count() -> tuple[int, str]:
    """Return physical CPU core count and the detection method."""
    try:
        import psutil

        count = psutil.cpu_count(logical=False)
        if count:
            return int(count), "physical cores (psutil)"
    except ImportError:
        pass
    return max(1, os.cpu_count() or 1), "os.cpu_count fallback (may be logical)"


def _initialize_worker(
    digits_name: str,
    digit_count: int,
    positions_name: str,
) -> None:
    """Attach each spawned process to the shared input and output arrays."""
    global _WORKER_DIGITS_SHM, _WORKER_POSITIONS_SHM
    global _WORKER_DIGITS, _WORKER_POSITIONS

    _WORKER_DIGITS_SHM = SharedMemory(name=digits_name)
    _WORKER_POSITIONS_SHM = SharedMemory(name=positions_name)
    _WORKER_DIGITS = np.ndarray(
        (digit_count,), dtype=np.uint8, buffer=_WORKER_DIGITS_SHM.buf
    )
    _WORKER_POSITIONS = np.ndarray(
        (digit_count, 2), dtype=np.float32, buffer=_WORKER_POSITIONS_SHM.buf
    )


def _summarize_local_block(
    task: tuple[int, int, int],
) -> tuple[int, int, float, float, int]:
    """Count local movement directions and return this block's displacement."""
    if _WORKER_DIGITS is None:
        raise RuntimeError("Worker shared-memory digits were not initialized.")

    start, stop, chunk_size = task
    angle_state = 0
    direction_counts = np.zeros(10, dtype=np.int64)

    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop)
        angle_units = np.cumsum(
            _WORKER_DIGITS[chunk_start:chunk_stop], dtype=np.uint32
        )
        angle_units += angle_state
        np.remainder(angle_units, 10, out=angle_units)
        direction_counts += np.bincount(angle_units, minlength=10)
        angle_state = int(angle_units[-1])

    local_dx = float(np.dot(direction_counts, _STEP_X))
    local_dy = float(np.dot(direction_counts, _STEP_Y))
    return start, stop, local_dx, local_dy, angle_state


def _calculate_global_block(
    task: tuple[int, int, int, int, float, float],
) -> tuple[int, int]:
    """Calculate final global coordinates for one block into shared memory."""
    if _WORKER_DIGITS is None or _WORKER_POSITIONS is None:
        raise RuntimeError("Worker shared-memory arrays were not initialized.")

    start, stop, chunk_size, angle_state, x_state, y_state = task

    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop)
        chunk_digits = _WORKER_DIGITS[chunk_start:chunk_stop]

        angle_units = np.cumsum(chunk_digits, dtype=np.uint32)
        angle_units += angle_state
        np.remainder(angle_units, 10, out=angle_units)

        step_x = _STEP_X[angle_units]
        step_y = _STEP_Y[angle_units]
        local_x = np.cumsum(step_x, dtype=np.float64)
        local_y = np.cumsum(step_y, dtype=np.float64)

        _WORKER_POSITIONS[chunk_start:chunk_stop, 0] = x_state + local_x
        _WORKER_POSITIONS[chunk_start:chunk_stop, 1] = y_state + local_y

        x_state += float(local_x[-1])
        y_state += float(local_y[-1])
        angle_state = int(angle_units[-1])

    return start, stop


def _build_spatial_bounds(
    positions: np.ndarray,
    digit_count: int,
    index_chunk_points: int,
    calculation_chunk_points: int,
) -> np.ndarray:
    """Build the same fixed-size spatial index used by the sequential path."""
    index_count = (digit_count + index_chunk_points - 1) // index_chunk_points
    bounds = np.empty((index_count, 4), dtype=np.float32)
    bounds[:, 0] = np.inf
    bounds[:, 1] = -np.inf
    bounds[:, 2] = np.inf
    bounds[:, 3] = -np.inf
    bounds[0] = (0.0, 0.0, 0.0, 0.0)

    for chunk_start in range(1, digit_count, calculation_chunk_points):
        chunk_stop = min(chunk_start + calculation_chunk_points, digit_count)
        first_index = chunk_start // index_chunk_points
        last_index = (chunk_stop - 1) // index_chunk_points
        for index in range(first_index, last_index + 1):
            box_start = max(chunk_start, index * index_chunk_points)
            box_stop = min(chunk_stop, (index + 1) * index_chunk_points)
            part = positions[box_start:box_stop]
            box = bounds[index]
            box[0] = min(box[0], float(part[:, 0].min()))
            box[1] = max(box[1], float(part[:, 0].max()))
            box[2] = min(box[2], float(part[:, 1].min()))
            box[3] = max(box[3], float(part[:, 1].max()))

    if not np.all(np.isfinite(bounds)):
        raise RuntimeError("Parallel spatial index contains uninitialized blocks.")
    return bounds


def _cleanup_parent_shared_memory() -> None:
    """Unlink parent-owned output segments during interpreter shutdown."""
    for handle in _PARENT_OUTPUT_HANDLES:
        try:
            handle.unlink()
        except (FileNotFoundError, OSError):
            pass
        try:
            handle.close()
        except (BufferError, OSError):
            # NumPy can still hold an exported view. The OS releases the mapping
            # when the application process exits.
            pass


atexit.register(_cleanup_parent_shared_memory)


def full_trajectory_parallel(
    digits: np.ndarray,
    *,
    worker_count: int,
    chunk_size: int,
    index_chunk_points: int,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the full trajectory in spawned workers using shared memory.

    The digit array is copied once into shared memory so Windows workers do
    not serialize a multi-gigabyte NumPy array. Workers write final float32 XY
    coordinates directly into their disjoint output ranges. The returned
    positions view remains backed by a parent-owned shared-memory mapping.
    """
    digit_count = int(digits.size)
    if digit_count < 2:
        raise ValueError("At least two digits are required.")
    if worker_count < 2:
        raise ValueError("Parallel trajectory generation requires 2+ workers.")
    if chunk_size < 1 or index_chunk_points < 1:
        raise ValueError("Chunk sizes must be positive.")

    movement_count = digit_count - 1
    worker_count = min(int(worker_count), movement_count)
    boundaries = [
        1 + (movement_count * worker_index) // worker_count
        for worker_index in range(worker_count + 1)
    ]
    block_ranges = [
        (boundaries[i], boundaries[i + 1])
        for i in range(worker_count)
        if boundaries[i] < boundaries[i + 1]
    ]
    summary_tasks = [
        (start, stop, chunk_size) for start, stop in block_ranges
    ]

    input_shm: SharedMemory | None = None
    output_shm: SharedMemory | None = None
    started = perf_counter()
    try:
        input_shm = SharedMemory(create=True, size=digits.nbytes)
        output_shm = SharedMemory(
            create=True,
            size=digit_count * 2 * np.dtype(np.float32).itemsize,
        )
        shared_digits = np.ndarray(
            (digit_count,), dtype=np.uint8, buffer=input_shm.buf
        )
        shared_digits[:] = digits
        positions = np.ndarray(
            (digit_count, 2), dtype=np.float32, buffer=output_shm.buf
        )
        positions[0] = (0.0, 0.0)

        if progress:
            print(
                f"  Multiprocessing: {worker_count} workers, "
                f"about {movement_count / worker_count:,.0f} digits per worker; "
                "shared-memory input/output"
            )

        context = mp.get_context("spawn")
        with context.Pool(
            processes=worker_count,
            initializer=_initialize_worker,
            initargs=(input_shm.name, digit_count, output_shm.name),
        ) as pool:
            summaries: list[tuple[int, int, float, float, int]] = []
            for completed, summary in enumerate(
                pool.imap_unordered(_summarize_local_block, summary_tasks),
                start=1,
            ):
                summaries.append(summary)
                if progress:
                    print(
                        f"  Block summaries: {completed}/{worker_count} complete"
                    )

            # Compose each block's correct global starting angle and coordinate.
            summaries.sort(key=lambda item: item[0])
            global_angle = int(digits[0]) % 10
            global_x = 0.0
            global_y = 0.0
            position_tasks: list[tuple[int, int, int, int, float, float]] = []

            for start, stop, local_dx, local_dy, angle_delta in summaries:
                position_tasks.append(
                    (
                        start,
                        stop,
                        chunk_size,
                        global_angle,
                        global_x,
                        global_y,
                    )
                )
                cosine = float(_STEP_X[global_angle])
                sine = float(_STEP_Y[global_angle])
                global_x += local_dx * cosine - local_dy * sine
                global_y += local_dx * sine + local_dy * cosine
                global_angle = (global_angle + angle_delta) % 10

            if progress:
                print("  Calculating final global coordinates in parallel...")
            for completed, _ in enumerate(
                pool.imap_unordered(_calculate_global_block, position_tasks),
                start=1,
            ):
                if progress:
                    print(
                        f"  Coordinate blocks: {completed}/{worker_count} complete"
                    )

        del shared_digits
        input_shm.close()
        input_shm.unlink()
        input_shm = None

        if progress:
            print("  Building spatial index...")
        bounds = _build_spatial_bounds(
            positions,
            digit_count,
            index_chunk_points,
            chunk_size,
        )

        # Unlink the shared-memory name but keep the parent mapping alive for
        # the returned NumPy view. Windows releases it when this handle closes.
        output_shm.unlink()
        _PARENT_OUTPUT_HANDLES.append(output_shm)
        output_shm = None

        if progress:
            print(f"  Parallel trajectory complete in {perf_counter() - started:.1f} s")
        return positions, bounds

    except Exception:
        if input_shm is not None:
            try:
                input_shm.close()
            except (BufferError, OSError):
                pass
            try:
                input_shm.unlink()
            except FileNotFoundError:
                pass
        if output_shm is not None:
            try:
                output_shm.close()
            except (BufferError, OSError):
                pass
            try:
                output_shm.unlink()
            except FileNotFoundError:
                pass
        raise
