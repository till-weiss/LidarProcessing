import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import core.icp_alignment as icp


class TestICPAlignment(unittest.TestCase):
    def test_dz_consistency_rule(self):
        self.assertTrue(icp.evaluate_dz_consistency(0.8, 0.7, 1.0))
        self.assertFalse(icp.evaluate_dz_consistency(3.0, 0.7, 1.0))

    def test_apply_transform_xyz(self):
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=float)
        T = np.eye(4)
        T[:3, 3] = [0.5, -0.25, 0.7]
        out = icp.apply_transform_xyz(xyz, T)
        np.testing.assert_allclose(out, xyz + np.array([0.5, -0.25, 0.7]), atol=1e-9)

    def test_synthetic_icp_translation(self):
        if icp.o3d is None:
            self.skipTest("Open3D unavailable")
        rng = np.random.default_rng(42)
        xy = rng.uniform(-50, 50, size=(5000, 2))
        z = 0.02 * xy[:, 0] + 0.01 * xy[:, 1]
        target = np.column_stack([xy, z])

        shift = np.array([0.5, -0.3, 0.7])
        source = target - shift

        reg, T = icp.run_icp(source, target, voxel_size=1.0, max_dist=2.0, max_iters=60)
        m = icp.metrics_from_T(T)
        self.assertGreater(reg.fitness, 0.8)
        self.assertAlmostEqual(m["dz"], shift[2], delta=0.5)

        source_aligned = icp.apply_transform_xyz(source, T)
        dz_post = float(np.median(source_aligned[:, 2]) - np.median(target[:, 2]))
        self.assertAlmostEqual(dz_post, 0.0, delta=0.5)

    def test_regression_json_pattern(self):
        sample = {
            "dz_median_pre": 0.67,
            "dz": 11.72,
            "accepted": False,
            "rejected_by": ["max_abs_dz"]
        }
        threshold = 1.5
        if not icp.evaluate_dz_consistency(sample["dz"], sample["dz_median_pre"], threshold):
            sample["rejected_by"].append("dz_inconsistent_with_precheck")
        self.assertIn("dz_inconsistent_with_precheck", sample["rejected_by"])


if __name__ == "__main__":
    unittest.main()
