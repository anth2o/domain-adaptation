from models.test_grl import TestGRL
import numpy as np

model = TestGRL()
x = np.array([[1]])
_lambda = np.array([[0.5]])
y = np.array([[2]])
model._build()
model._compile()
previous_weight = model.model.layers[1].get_weights()[0][0][0]
model._fit([x, _lambda], y)
print(previous_weight - _lambda[0][0] * model.lr * (y - x)[0][0] - model.model.layers[1].get_weights()[0][0][0] < 10e-5)
