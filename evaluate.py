from models.cnn import CNN
from models.cnn_grl import CNNGRL
from utils.preprocessing import Preprocessor
from utils.config import *

pp = Preprocessor(domains_ignore_labels=['svhn'])
((x_train, y_train), (x_test, y_test)), ((x_train_unlabelled, y_train_unlabelled), (x_test_unlabelled, y_test_unlabelled)) = pp.get_data()

path_to_model = {
    'weights/cnn_grl_train_svhn.h5': CNNGRL,
    'weights/cnn_train_svhn.h5': CNN
}
for path, model_class in path_to_model.items():
    model = model_class()
    print('Evaluating model {} with weights {}'.format(model_class.__name__, path))
    model._load_and_evaluate(path, x_test, y_test['label'])
