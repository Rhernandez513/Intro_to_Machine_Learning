import unittest
from hw1 import prime_nums_reversed



class TestQ01A(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_nums_one(self):
        """Evaluate prime_nums_reversed(1)"""
        self.assertEqual(prime_nums_reversed(1), '')

    def test_nums_five(self):
        """Evaluate prime_nums_reversed(5)"""
        self.assertEqual(prime_nums_reversed(5), [5, 3, 2])

    def test_nums_nine(self):
        """Evaluate prime_nums_reversed(5)"""
        self.assertEqual(prime_nums_reversed(9), [9, 7, 5, 3, 2])
