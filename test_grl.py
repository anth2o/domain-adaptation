from models.test_grl import TestGRL
import numpy as np

model = TestGRL()
x = np.array([[1]])
y = np.array([[2]])
model._build()
model._compile()
previous_weight = model.model.layers[1].get_weights()[0][0][0]
model._fit(x, y)
print(previous_weight - model._lambda * model.lr * (y - x)[0][0] - model.model.layers[1].get_weights()[0][0][0] < 10e-5)
