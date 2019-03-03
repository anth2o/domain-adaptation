import unittest
import numpy as np
from models.simple_grl import SimpleGRL

class TestGRL(unittest.TestCase):
    def test_value(self):
        model = SimpleGRL()
        x = np.array([[1]])
        _lambda = np.array([[0.5]])
        y = np.array([[2]])
        model._build()
        model._compile()
        previous_weight = model.model.layers[1].get_weights()[0][0][0]
        model._fit([x, _lambda], y)
        self.assertAlmostEqual(previous_weight - _lambda[0][0] * model.lr * (y - x)[0][0], model.model.layers[1].get_weights()[0][0][0])

if __name__ == '__main__':
    unittest.main()
