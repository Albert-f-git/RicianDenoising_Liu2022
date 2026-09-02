import unittest

import numpy as np

from src.operators import compute_adaptive_g
from src.solvers import RicianSolver


class SolverTests(unittest.TestCase):
    def test_adaptive_estimate_uses_paper_default(self):
        f = np.array([[10.0]])
        np.testing.assert_allclose(compute_adaptive_g(f, sigma=2.0), [[np.sqrt(92.0)]])

    def test_sphere_projection_satisfies_constraint_at_zero(self):
        solver = RicianSolver(alpha=0.01, beta=0.045, sigma=25.0)
        f = np.array([[3.0, 4.0]])
        zeros = np.zeros_like(f)
        v1, v2 = solver._update_v(zeros, zeros, zeros, f)
        np.testing.assert_allclose(np.hypot(v1, v2), f)

    def test_rof_center_matches_equation_4_12_for_constants(self):
        solver = RicianSolver(alpha=0.2, beta=0.5, r=2.0, sigma=1.0)
        g = np.full((4, 5), 3.0)
        v1 = np.full_like(g, 7.0)
        n1 = np.full_like(g, 2.0)
        expected = (solver.beta * g + solver.r * v1 + (solver.alpha - solver.r) * n1) / (solver.beta + solver.r)
        np.testing.assert_allclose(solver._update_u(v1, n1, g), expected)


if __name__ == "__main__":
    unittest.main()
