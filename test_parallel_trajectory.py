from __future__ import annotations

import unittest

import numpy as np

import pi_visualize as app
from pi_trajectory_mp import full_trajectory_parallel


class ParallelTrajectoryTests(unittest.TestCase):
    def _compare(self, digits: np.ndarray, workers: int) -> None:
        expected_positions, expected_bounds = app.full_trajectory_sequential(digits)
        actual_positions, actual_bounds = full_trajectory_parallel(
            digits,
            worker_count=workers,
            chunk_size=4096,
            index_chunk_points=997,
            progress=False,
        )

        error = np.max(np.abs(actual_positions - expected_positions))
        self.assertLessEqual(float(error), 0.0015)
        self.assertEqual(actual_positions.shape, expected_positions.shape)
        self.assertTrue(np.all(np.isfinite(actual_bounds)))

        # Recompute all index boxes directly from the returned positions.
        for index, bounds in enumerate(actual_bounds):
            start = index * 997
            stop = min(start + 997, len(actual_positions))
            part = actual_positions[start:stop]
            reference = np.array(
                [
                    part[:, 0].min(),
                    part[:, 0].max(),
                    part[:, 1].min(),
                    part[:, 1].max(),
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(bounds, reference, rtol=0, atol=1e-5)

    def test_parallel_matches_sequential_for_odd_block_boundaries(self) -> None:
        rng = np.random.default_rng(20261112)
        digits = rng.integers(0, 10, size=31_337, dtype=np.uint8)
        self._compare(digits, workers=3)

    def test_parallel_matches_sequential_when_blocks_are_not_evenly_divisible(self) -> None:
        rng = np.random.default_rng(42)
        digits = rng.integers(0, 10, size=57_913, dtype=np.uint8)
        self._compare(digits, workers=4)

    def test_initial_digit_only_sets_direction(self) -> None:
        digits = np.array([3, 0, 0, 0, 0, 0], dtype=np.uint8)
        positions, _ = full_trajectory_parallel(
            digits,
            worker_count=2,
            chunk_size=2,
            index_chunk_points=2,
            progress=False,
        )
        self.assertEqual(tuple(positions[0]), (0.0, 0.0))
        self.assertLess(float(positions[1, 0]), 0.0)
        self.assertGreater(float(positions[1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
