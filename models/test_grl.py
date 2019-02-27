from .base_model import BaseModel
from layers.grl import GRL
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.models import Model


class TestGRL(BaseModel):
    def __init__(self):
        super(TestGRL, self).__init__()
        self.loss = 'mae'
        self.lr = 0.1
        self.opt = SGD(lr=self.lr)
        self._lambda = 0.5

    def _build(self):
        inputs = Input(shape=(1,))
        x = Dense(1, use_bias=False)(inputs)
        predictions = GRL(_lambda=self._lambda)(x)
        self.model = Model(inputs=inputs, outputs=predictions)

    def _fit(self, x_train, y_train, epochs=1):
        if not self.model:
            raise Exception("Trying to fit model but it isn't built")
        self.model.fit(x_train, y_train, epochs=1)