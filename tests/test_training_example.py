"""Regression checks for the runnable training example, not the generator engine."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("training_example", Path(__file__).resolve().parents[1] / "examples" / "training_quickstart.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class TrainingExampleTests(unittest.TestCase):
    def test_evaluation_outliers_do_not_change_training_normalization(self):
        data = np.array([[0., 2.], [2., 6.], [1000., -1000.]])
        xt, yt, xe, ye = example.prepare(data, 1, 2)
        np.testing.assert_allclose(xt.numpy().ravel(), [-1., 1.])
        np.testing.assert_allclose(yt.numpy().ravel(), [-1., 1.])
        np.testing.assert_allclose(xe.numpy().ravel(), [999.])
        np.testing.assert_allclose(ye.numpy().ravel(), [-502.])

    def test_real_update_reproducible_metrics(self):
        a = example.run(rows=32, steps=10, seed=11)
        b = example.run(rows=32, steps=10, seed=11)
        self.assertTrue(a["parameters_changed"])
        self.assertEqual(a, b)
        self.assertEqual(a["train_rows"], 24)
        self.assertEqual(a["eval_rows"], 8)


if __name__ == "__main__":
    unittest.main()
