import unittest
from hw1 import replace


class TestQ01B(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_replace_simple(self):
        """Evaluate replace('[1, 2, 3, 4], [5, 6, 7, 8]')"""
        self.assertEqual(replace([1, 2, 3, 4], [5, 6, 7, 8]), [1, 2, 3, 5, 6, 7, 8])

