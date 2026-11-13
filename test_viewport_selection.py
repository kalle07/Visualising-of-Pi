"""Small regression test for viewport point selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("pi_visualize.py")
spec = importlib.util.spec_from_file_location("pi_visualize", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
pi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pi)


def make_bounds(positions: np.ndarray, chunk_size: int) -> np.ndarray:
    count = (len(positions) + chunk_size - 1) // chunk_size
    bounds = np.empty((count, 4), dtype=np.float32)
    for index in range(count):
        block = positions[index * chunk_size : (index + 1) * chunk_size]
        bounds[index] = (
            block[:, 0].min(), block[:, 0].max(),
            block[:, 1].min(), block[:, 1].max(),
        )
    return bounds


def test_selected_indices_match_direct_reference() -> None:
    rng = np.random.default_rng(2026)
    pi.VIEW_INDEX_CHUNK_POINTS = 37

    for point_count in (1, 36, 37, 38, 111, 1_000):
        positions = rng.uniform(-100, 100, (point_count, 2)).astype(np.float32)
        bounds = make_bounds(positions, pi.VIEW_INDEX_CHUNK_POINTS)
        rectangle = (-20.0, 25.0, -33.0, 31.0)
        mask = (
            (positions[:, 0] >= rectangle[0])
            & (positions[:, 0] <= rectangle[1])
            & (positions[:, 1] >= rectangle[2])
            & (positions[:, 1] <= rectangle[3])
        )
        reference = np.flatnonzero(mask)

        for budget in (1, 3, 10, 10_000):
            actual = pi.visible_point_indices(
                positions, bounds, rectangle, budget
            )
            if len(reference) <= budget:
                np.testing.assert_array_equal(actual, reference)
            else:
                ranks = np.linspace(
                    0, len(reference) - 1, budget, dtype=np.int64
                )
                np.testing.assert_array_equal(actual, reference[ranks])


def test_zoom_budget_is_capped() -> None:
    assert pi.dynamic_point_budget(500.0, 500.0, 100_000_000) == 2_000_000
    assert pi.MAX_GPU_POINTS == 2_000_000


def test_upload_cap_is_below_50_mib() -> None:
    upload_bytes = pi.MAX_GPU_POINTS * 24
    assert upload_bytes < 50 * 1024 * 1024


if __name__ == "__main__":
    test_selected_indices_match_direct_reference()
    test_zoom_budget_is_capped()
    test_upload_cap_is_below_50_mib()
    print("3 viewport regression tests passed.")
