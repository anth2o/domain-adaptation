import keras
from scipy.io import loadmat
import numpy as np

def get_data(num_classes, subset=False, num_domains=2, is_svhn=True):
    # TODO: read MNIST data set and init y_domain correctly
    if is_svhn:
        folder = 'data/svhn/'
        train, test = loadmat(folder + 'train.mat'), loadmat(folder + 'test.mat')
        x_train, y_train_label = read_svhn(train, subset)
        x_test, y_test_label = read_svhn(test, subset)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = process_x(x_train)
    x_test = process_x(x_test)

    y_train_domain = np.ones_like(y_train_label)
    y_test_domain = np.ones_like(y_test_label)

    y_train_label = process_y(y_train_label, num_classes)
    y_test_label = process_y(y_test_label, num_classes)

    y_train_domain = process_y(y_train_domain, num_domains)
    y_test_domain = process_y(y_test_domain, num_domains)

    return (x_train, [y_train_label, y_train_domain]), (x_test, [y_test_label, y_test_domain])

def process_x(x):
    x = x.astype('float32')
    x /= 255
    return x

def process_y(y, num_classes):
    return keras.utils.to_categorical(y, num_classes)

def read_svhn(dataset, subset=False):
    x = dataset['X']
    x = np.moveaxis(x, -1, 0)
    y = dataset['y']
    y[y==10] = 0
    if subset:
        x = x[:subset]
        y = y[:subset]
    return x, y
