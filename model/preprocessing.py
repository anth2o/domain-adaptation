import keras
from scipy.io import loadmat
import numpy as np

def get_data(num_classes, subset=False, is_svhn=True):
    if is_svhn:
        folder = 'data/svhn/'
        train, test = loadmat(folder + 'train.mat'), loadmat(folder + 'test.mat')
        x_train, y_train = process_svhn(train, subset)
        x_test, y_test = process_svhn(test, subset)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def process_svhn(dataset, subset=False):
    x = dataset['X']
    x = np.moveaxis(x, -1, 0)
    y = dataset['y']
    y[y==10] = 0
    if subset:
        x = x[:subset]
        y = y[:subset]
    return x, y
