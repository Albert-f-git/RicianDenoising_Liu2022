import unittest

import numpy as np

from src.operators import backward_divergence, forward_gradient


class OperatorTests(unittest.TestCase):
    def test_gradient_and_divergence_are_negative_adjoints(self):
        rng = np.random.default_rng(123)
        u = rng.standard_normal((31, 37))
        px = rng.standard_normal(u.shape)
        py = rng.standard_normal(u.shape)
        gx, gy = forward_gradient(u)
        lhs = np.sum(gx * px + gy * py)
        rhs = np.sum(u * (-backward_divergence(px, py)))
        self.assertAlmostEqual(float(lhs), float(rhs), places=10)


if __name__ == "__main__":
    unittest.main()
