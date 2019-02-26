import keras
from scipy.io import loadmat
import numpy as np
import gzip
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize
from skimage.color import gray2rgb

def get_data(num_classes, image_size, channels, domains, subset=False):
    num_domains = len(domains)
    x_train_list = []
    y_train_label_list = []
    y_train_domain_list = []
    x_test_list = []
    y_test_label_list = []
    y_test_domain_list = []
    for i in range(num_domains):
        (x_train, y_train_label), (x_test, y_test_label) = get_single_data(num_classes, image_size, channels, num_domains, subset, domain=domains[i])
        y_train_domain = get_y_domain(y_train_label, num_domains, domain_value=i)
        y_test_domain = get_y_domain(y_test_label, num_domains, domain_value=i)

        y_train_label = process_y(y_train_label, num_classes)
        y_test_label = process_y(y_test_label, num_classes)

        x_train_list.append(x_train)
        y_train_label_list.append(y_train_label)
        y_train_domain_list.append(y_train_domain)

        x_test_list.append(x_test)
        y_test_label_list.append(y_test_label)
        y_test_domain_list.append(y_test_domain)

    x_train = np.concatenate(x_train_list, axis=0)
    y_train_label = np.concatenate(y_train_label_list, axis=0)
    y_train_domain = np.concatenate(y_train_domain_list, axis=0)

    x_test = np.concatenate(x_test_list, axis=0)
    y_test_label = np.concatenate(y_test_label_list, axis=0)
    y_test_domain = np.concatenate(y_test_domain_list, axis=0)

    return (x_train, [y_train_label, y_train_domain]), (x_test, [y_test_label, y_test_domain])


def get_single_data(num_classes, image_size, channels, num_domains=2, subset=False, domain='svhn'):
    if domain=='svhn':
        folder = 'data/svhn/'
        train, test = loadmat(folder + 'train.mat'), loadmat(folder + 'test.mat')
        x_train, y_train_label = read_svhn(train)
        x_test, y_test_label = read_svhn(test)
    elif domain == 'mnist':
        (x_train, y_train_label),(x_test, y_test_label) = tf.keras.datasets.mnist.load_data()    

    if subset:
        x_train = x_train[:subset]
        y_train_label = y_train_label[:subset]
        x_test = x_test[:subset]
        y_test_label = y_test_label[:subset]   

    print('x_train {} shape: {}'.format(domain, x_train.shape))
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = process_x(x_train, image_size, channels)
    x_test = process_x(x_test, image_size, channels)
    
    return (x_train, y_train_label), (x_test, y_test_label)

def process_x(x, image_size, channels, subset=False):
    x = x.astype('float32')
    # Resize the images to a common size
    if x.shape[1:3] != image_size:
        x = np.moveaxis(x, 0, -1)
        x = resize(x, image_size, anti_aliasing=True, mode='constant')
        x = np.moveaxis(x, -1, 0)
    # Convert single channel to 3 channels picture
    if len(x.shape) < 4 or x.shape[3] == 1:
        x = gray2rgb(x)
    x /= 255
    return x

def process_y(y, num_classes):
    y = keras.utils.to_categorical(y, num_classes)
    return y

def get_y_domain(y_label, num_domains, domain_value):
    y_domain = np.ones_like(y_label) * domain_value
    return process_y(y_domain, num_domains)

def read_svhn(dataset):
    x = dataset['X']
    x = np.moveaxis(x, -1, 0)
    y = dataset['y']
    y[y==10] = 0
    return x, y
