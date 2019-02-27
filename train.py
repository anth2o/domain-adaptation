import numpy as np

from models.cnn import CNN
from models.cnn_grl import CNNGRL
from utils.preprocessing import Preprocessor
from utils.config import *

pp = Preprocessor()
((x_train, y_train), (x_test, y_test)), ((x_train_unlabelled, y_train_unlabelled), (x_test_unlabelled, y_test_unlabelled)) = pp.get_data()

architecture = 'CNNGRL'

if architecture == 'CNN':
    model = CNN()
    model._run_all(x_train, x_test, y_train['label'], y_test['label'])
elif architecture == 'CNNGRL':
    model = CNNGRL()
    model._run_all(x_train, x_test, y_train, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled)
