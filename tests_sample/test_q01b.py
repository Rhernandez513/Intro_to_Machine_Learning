import unittest
from hw1 import string_explosion


class TestQ01B(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_string_explosion_code(self):
        """Evaluate string_explosion('Code')"""
        self.assertEqual(string_explosion("Code"), 'Codeodedee')

    def test_string_explosion_hi(self):
        """Evaluate string_explosion('Hi')"""
        self.assertEqual(string_explosion("Hi"), 'Hii')

    def test_string_explosion_data(self):
        """Evaluate string_explosion('data!')"""
        self.assertEqual(string_explosion("data!"), 'data!ata!ta!a!!')



