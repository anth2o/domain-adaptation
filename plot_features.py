from models.cnn import CNN
from models.cnn_grl import CNNGRL
from utils.preprocessing import Preprocessor
from utils.config import *
from keras.models import Model
from sklearn.manifold import t_sne
import matplotlib.pyplot as plt
import os

subset = 3000
pp = Preprocessor(subset=subset)
(x_train_svhn, y_train_svhn), (x_test_svhn, y_test_svhn) = pp.get_one_domain_data('svhn')
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = pp.get_one_domain_data('mnist')

def plot_features(model_class=CNNGRL, model_weights='cnn_grl_train_svhn', subset=subset):
    model = model_class()
    model._build()
    model._load_weights(model_weights)
    
    if model_class==CNNGRL:
        model_input = model.model_label.input
        model_output = model.model_label.get_layer('dropout_label_1').output
    else:
        model_input = model.model.input
        model_output = model.model.layers[-3].output

    intermediate_layer_model = Model(inputs=model_input, outputs=model_output)
    features_svhn = intermediate_layer_model.predict(x_train_svhn)
    features_mnist = intermediate_layer_model.predict(x_train_mnist)

    tsne_svhn = t_sne.TSNE().fit_transform(features_svhn)
    tsne_mnist = t_sne.TSNE().fit_transform(features_mnist)

    plt.close()
    plt.scatter(tsne_svhn[:, 0], tsne_svhn[:, 1], color='b', s=1, label='svhn')
    plt.scatter(tsne_mnist[:, 0], tsne_mnist[:, 1], color='r', s=1, label='mnist')
    plt.title(model_weights)
    plt.legend()

    if subset is None:
        save_dir = 'img/tsne_features/'
    else:
        save_dir = 'img/tsne_features_' + str(subset) + '/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + model_weights + '.png')
    print('Figure saved at ' + save_dir + model_weights + '.png')

plot_features()
plot_features(CNN, 'cnn_train_svhn')
