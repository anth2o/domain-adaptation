from .base_model import BaseModel
from layers.grl import GRL
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.models import Model
import numpy as np


class SimpleGRL(BaseModel):
    def __init__(self):
        super(SimpleGRL, self).__init__()
        self.loss = 'mae'
        self.lr = 0.1
        self.opt = SGD(lr=self.lr)

    def _build(self):
        inputs_x = Input(shape=(1,))
        inputs_lambda = Input(shape=(1,))
        x = Dense(1, use_bias=False)(inputs_x)
        predictions = GRL()([x, inputs_lambda])
        self.model = Model(inputs=[inputs_x, inputs_lambda], outputs=predictions)

    def _fit(self, x_train, y_train, epochs=1):
        if not self.model:
            raise Exception("Trying to fit model but it isn't built")
        self.model.fit(x_train, y_train, epochs=1)
