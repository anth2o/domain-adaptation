from models.cnn import CNN
from models.cnn_grl import CNNGRL
from utils.preprocessing import Preprocessor
from utils.config import *
from sklearn.manifold import t_sne
from keras.models import Model
import matplotlib.pyplot as plt

pp = Preprocessor(subset=100)
(x_train_svhn, y_train_svhn), (x_test_svhn, y_test_svhn) = pp.get_one_domain_data('svhn')
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = pp.get_one_domain_data('mnist')

def plot_features(model_class=CNNGRL, model_weights='cnn_grl_train_svhn'):
    model = model_class()
    model._build()
    model._load_weights(model_weights)
    
    if model_class==CNNGRL:
        model = model.model_label
    else:
        model = model.model

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[-2].output)
    features_svhn = intermediate_layer_model.predict(x_train_svhn)
    features_mnist = intermediate_layer_model.predict(x_train_mnist)

    tsne_svhn = t_sne.TSNE().fit_transform(features_svhn)
    tsne_mnist = t_sne.TSNE().fit_transform(features_mnist)

    plt.scatter(tsne_svhn[:, 0], tsne_svhn[:, 1], color='b')
    plt.scatter(tsne_mnist[:, 0], tsne_mnist[:, 1], color='r')
    plt.show()

plot_features()
plot_features(CNN, 'cnn_train_svhn')
