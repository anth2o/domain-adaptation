from models.cnn import CNN
from models.cnn_grl import CNNGRL
from utils.preprocessing import Preprocessor
from utils.config import *

pp = Preprocessor(subset=None)
(x_train_svhn, y_train_svhn), (x_test_svhn, y_test_svhn) = pp.get_one_domain_data('svhn')
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = pp.get_one_domain_data('mnist')


path_to_model = {
    'weights/cnn_train_svhn.h5': CNN,
    'weights/cnn_grl_merge_train_svhn.h5': CNNGRL,
}
for path, model_class in path_to_model.items():
    try:
        model = model_class()
        print('')
        print('Evaluating model {} with weights {}'.format(model_class.__name__, path))
        print('Evaluating on SVHN test set')
        model._load_and_evaluate(path, x_test_svhn, y_test_svhn)
        print('Evaluating on MNIST test set')
        model._load_and_evaluate(path, x_test_mnist, y_test_mnist)
    except Exception as e:
        print(e)
