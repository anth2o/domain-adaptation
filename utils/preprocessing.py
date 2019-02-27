import keras
from scipy.io import loadmat
import numpy as np
import gzip
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize
from skimage.color import gray2rgb
from utils.config import *

class Preprocessor():
    def __init__(self, num_classes=NUM_CLASSES, domains=DOMAINS, domains_ignore_labels=DOMAINS_IGNORE_LABELS, image_size=IMAGE_SIZE, channels=CHANNELS, subset=SUBSET):
        self.num_classes = num_classes
        self.domains = domains
        self.num_domains = len(domains)
        self.domains_not_ignore_labels = list(set(domains) - set(domains_ignore_labels))
        self.image_size = image_size
        self.channels = channels
        self.subset = subset

    def get_data(self):
        """
        Returns:
        ((x_train, y_train), (x_test, y_test)), ((x_train_unlabelled, y_train_unlabelled), (x_test_unlabelled, y_test_unlabelled))
        With ((x_train, y_train), (x_test, y_test)) the data for the domains where the label is available
        And ((x_train_unlabelled, y_train_unlabelled), (x_test_unlabelled, y_test_unlabelled)) all the data but with no label data
        """
        self.get_dict_data()


        x_train = self.concatenate([x for x in [self.x_train_dict[key] for key in self.domains_not_ignore_labels]])
        y_train_label = self.concatenate([y for y in [self.y_train_label_dict[key] for key in self.domains_not_ignore_labels]])
        y_train_domain = self.concatenate([y for y in [self.y_train_domain_dict[key] for key in self.domains_not_ignore_labels]])
        y_train = {
            'label': y_train_label,
            'domain': y_train_domain
            }

        x_test = self.concatenate([x for x in [self.x_test_dict[key] for key in self.domains_not_ignore_labels]])
        y_test_label = self.concatenate([y for y in [self.y_test_label_dict[key] for key in self.domains_not_ignore_labels]])
        y_test_domain = self.concatenate([y for y in [self.y_test_domain_dict[key] for key in self.domains_not_ignore_labels]])
        y_test = {
            'label': y_test_label,
            'domain': y_test_domain
            }

        x_train_unlabelled = self.concatenate([x for x in [self.x_train_dict[key] for key in self.domains]])
        y_train_unlabelled = self.concatenate([y for y in [self.y_train_domain_dict[key] for key in self.domains]])
        y_train_unlabelled = {
            'domain': y_train_unlabelled
        }

        x_test_unlabelled = self.concatenate([x for x in [self.x_test_dict[key] for key in self.domains]])
        y_test_unlabelled = self.concatenate([y for y in [self.y_test_domain_dict[key] for key in self.domains]])
        y_test_unlabelled = {
            'domain': y_test_unlabelled
        }

        return ((x_train, y_train), (x_test, y_test)), ((x_train_unlabelled, y_test_unlabelled), (x_test_unlabelled, y_test_unlabelled))

    def get_dict_data(self):
        self.x_train_dict = {}
        self.y_train_label_dict = {}
        self.y_train_domain_dict = {}
        self.x_test_dict = {}
        self.y_test_label_dict = {}
        self.y_test_domain_dict = {}
        for i in range(self.num_domains):
            domain = self.domains[i]
            (x_train, y_train_label), (x_test, y_test_label) = self.get_one_domain_data(domain=domain)
            y_train_domain = self.get_y_domain(y_train_label, domain_value=i)
            y_test_domain = self.get_y_domain(y_test_label, domain_value=i)

            y_train_label = self.process_y(y_train_label, self.num_classes)
            y_test_label = self.process_y(y_test_label, self.num_classes)

            self.x_train_dict[domain] = x_train
            self.y_train_label_dict[domain] = y_train_label
            self.y_train_domain_dict[domain] = y_train_domain

            self.x_test_dict[domain] = x_test
            self.y_test_label_dict[domain] = y_test_label
            self.y_test_domain_dict[domain] = y_test_domain

    def concatenate(self, _list):
        if len(_list) > 0:
            return np.concatenate(_list, axis=0)
        return np.array([])


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
