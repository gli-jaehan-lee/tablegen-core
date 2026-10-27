"""0.2.0 defaults: cheap restriction auto, robust 4-sigma clamp, redraw of unusable tables."""
import unittest
from unittest import mock
import numpy as np
import tablegen_core.api as api
from tablegen_core import generate
from tablegen_core.postprocess import outlier_clip, all_features_constant


class DefaultsTest(unittest.TestCase):
    def test_cheap_restriction_auto(self):
        self.assertTrue(generate(mode="dag", priors=["default8"], rows=64, features=3, seed=1).metadata["config"]["restrict_cheap_functions"])
        self.assertFalse(generate(mode="direct", priors=["lin"], rows=64, features=3, seed=1).metadata["config"]["restrict_cheap_functions"])
        self.assertFalse(generate(mode="dag", priors=["robust16"], rows=64, features=3, seed=1).metadata["config"]["restrict_cheap_functions"])

    def test_clip_bounds_and_disable(self):
        x = np.zeros((200, 2), dtype=np.float32); x[:, 0] = np.linspace(-1, 1, 200); x[0, 0] = 1e6
        y, n = outlier_clip(x, 4.0)
        self.assertEqual(n, 1); self.assertLess(y[0, 0], 10)
        r = generate(mode="dag", priors=["all24"], rows=128, features=4, seed=3, outlier_clip=None)
        self.assertEqual(r.metadata["outlier_clip"]["clipped_cells"], 0)

    def test_categorical_target_not_clipped(self):
        r = generate(mode="dag", priors=["all24"], rows=128, features=4, seed=42, target_categories=[3])
        self.assertTrue(set(np.unique(r.data[:, -1])) <= {0.0, 1.0, 2.0})

    def test_redraw_on_nonfinite_and_constant(self):
        real = api._generate_once
        calls = {"n": 0}
        def flaky(cfg, **kw):
            calls["n"] += 1
            if calls["n"] == 1: raise api.NonfiniteOutputError("x")
            r = real(cfg, **kw)
            if calls["n"] == 2: r.data[:, :cfg.features] = 0.0
            return r
        with mock.patch.object(api, "_generate_once", flaky):
            r = generate(mode="dag", priors=["default8"], rows=64, features=3, seed=5)
        self.assertEqual(r.metadata["table_attempts"]["used"], 3)
        self.assertEqual([x["reason"] for x in r.metadata["table_attempts"]["rejected"]], ["NonfiniteOutputError", "all_inputs_constant"])
        self.assertFalse(all_features_constant(r.data, 3))
        with mock.patch.object(api, "_generate_once", side_effect=api.NonfiniteOutputError("x")):
            with self.assertRaises(api.NonfiniteOutputError):
                generate(mode="dag", priors=["default8"], rows=64, features=3, seed=5, max_table_attempts=1)

    def test_deterministic(self):
        a = generate(mode="dag", priors=["all24"], rows=96, features=5, seed=11)
        b = generate(mode="dag", priors=["all24"], rows=96, features=5, seed=11)
        self.assertTrue(np.array_equal(a.data, b.data))


if __name__ == "__main__":
    unittest.main()
