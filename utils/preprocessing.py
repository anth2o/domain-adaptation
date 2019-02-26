import keras
from scipy.io import loadmat
import numpy as np
import gzip
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize
from skimage.color import gray2rgb

class Preprocessor():
    def __init__(self, num_classes, domains, ignore_labels, image_size, channels, subset):
        self.num_classes = num_classes
        self.domains = domains
        self.num_domains = len(domains)
        self.ignore_labels = ignore_labels
        self.not_ignore_labels = list(set(domains) - set(ignore_labels))
        self.image_size = image_size
        self.channels = channels
        self.subset = subset

    def get_data(self):
       return self.get_one_type_data(ignore=False), self.get_one_type_data(ignore=True)

    def get_one_type_data(self, ignore):
        x_train_list = []
        y_train_label_list = []
        y_train_domain_list = []
        x_test_list = []
        y_test_label_list = []
        y_test_domain_list = []
        if ignore:
            local_domains = self.ignore_labels
        else:
            local_domains = self.not_ignore_labels
        for i in range(len(local_domains)):
            (x_train, y_train_label), (x_test, y_test_label) = self.get_one_domain_data(domain=local_domains[i])
            y_train_domain = self.get_y_domain(y_train_label, domain_value=ignore)
            y_test_domain = self.get_y_domain(y_test_label, domain_value=ignore)

            y_train_label = self.process_y(y_train_label, self.num_classes)
            y_test_label = self.process_y(y_test_label, self.num_classes)

            x_train_list.append(x_train)
            if not ignore:
                y_train_label_list.append(y_train_label)
            y_train_domain_list.append(y_train_domain)

            x_test_list.append(x_test)
            if not ignore:
                y_test_label_list.append(y_test_label)
            y_test_domain_list.append(y_test_domain)

        x_train = np.concatenate(x_train_list, axis=0)
        if not ignore:
            y_train_label = np.concatenate(y_train_label_list, axis=0)
        y_train_domain = np.concatenate(y_train_domain_list, axis=0)

        x_test = np.concatenate(x_test_list, axis=0)
        if not ignore:
            y_test_label = np.concatenate(y_test_label_list, axis=0)
        y_test_domain = np.concatenate(y_test_domain_list, axis=0)

        if ignore:
            return (x_train, [y_train_domain]), (x_test, [y_test_domain])
        return (x_train, [y_train_label, y_train_domain]), (x_test, [y_test_label, y_test_domain])


    def get_one_domain_data(self, domain='svhn'):
        if domain=='svhn':
            folder = 'data/svhn/'
            train, test = loadmat(folder + 'train.mat'), loadmat(folder + 'test.mat')
            x_train, y_train_label = self.read_svhn(train)
            x_test, y_test_label = self.read_svhn(test)
        elif domain == 'mnist':
            (x_train, y_train_label),(x_test, y_test_label) = tf.keras.datasets.mnist.load_data()    

        if self.subset:
            x_train = x_train[:self.subset]
            y_train_label = y_train_label[:self.subset]
            x_test = x_test[:self.subset]
            y_test_label = y_test_label[:self.subset]   

        print('x_train {} shape: {}'.format(domain, x_train.shape))
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        x_train = self.process_x(x_train)
        x_test = self.process_x(x_test)
        
        return (x_train, y_train_label), (x_test, y_test_label)

    def process_x(self, x):
        x = x.astype('float32')
        # Resize the images to a common size
        if x.shape[1:3] != self.image_size:
            x = np.moveaxis(x, 0, -1)
            x = resize(x, self.image_size, anti_aliasing=True, mode='constant')
            x = np.moveaxis(x, -1, 0)
        # Convert single channel to 3 channels picture
        if len(x.shape) < 4 or x.shape[3] == 1:
            x = gray2rgb(x)
        x /= 255
        return x

    def process_y(self, y, num_categories):
        y = keras.utils.to_categorical(y, num_categories)
        return y

    def get_y_domain(self, y_label, domain_value):
        y_domain = np.ones_like(y_label) * domain_value
        return self.process_y(y_domain, self.num_domains)

    def read_svhn(self, dataset):
        x = dataset['X']
        x = np.moveaxis(x, -1, 0)
        y = dataset['y']
        y[y==10] = 0
        return x, y
