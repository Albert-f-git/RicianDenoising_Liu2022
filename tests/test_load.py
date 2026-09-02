import unittest
from pathlib import Path

import numpy as np

from src.utils import load_brainweb_data


class BrainWebLoadTests(unittest.TestCase):
    def test_data_integrity(self):
        data_path = Path(__file__).resolve().parents[1] / "data" / "t1_icbm_normal_1mm_pn0_rf0.raws"
        image = load_brainweb_data(data_path)
        self.assertEqual(image.shape, (217, 181))
        self.assertTrue(np.isfinite(image).all())
        self.assertAlmostEqual(float(image.min()), 0.0, places=5)
        self.assertAlmostEqual(float(image.max()), 255.0, places=5)


if __name__ == "__main__":
    unittest.main()
