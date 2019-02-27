import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from uuid import uuid4

class GRL(Layer):
    def __init__(self, _lambda=1, **kwargs):
        self._lambda = _lambda
        super(GRL, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights = []
        super(GRL, self).build(input_shape)

    def call(self, x):
        grad_name = 'GRL-' + str(uuid4())

        @tf.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * self._lambda]

        g = K.get_session().graph
        with g.gradient_override_map({'Identity': grad_name}):
            y = tf.identity(x)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape
