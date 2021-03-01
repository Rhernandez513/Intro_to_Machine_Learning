import unittest

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover('tests_sample') # edit test_sample to add your own tests
    unittest.TextTestRunner().run(suite)
