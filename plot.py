from models.cnn import CNN
from models.cnn_grl import CNNGRL
from utils.config import *

path_to_model = {
    CNN_GRL_MODEL_NAME: CNNGRL,
    CNN_MODEL_NAME: CNN,
}

for path, model_class in path_to_model.items():
    model = model_class()
    model._build()
    model._plot_model()