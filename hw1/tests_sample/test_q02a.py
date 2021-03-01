import unittest
import numpy as np
from hw1 import bowl_cost


class TestQ02A(unittest.TestCase):
    def setUp(self):
        pass

    def test_one_three_five(self):
        """Evaluate bowl_cost([1, 3, 5])"""
        self.assertTrue(np.array_equal(bowl_cost([1, 3, 5]), np.array([36, 16, 22, 60])))
