import os
import os.path as op
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

folder = 'data/'

mnist_urls = {
    'train_images.tar.gz': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
}

svhn_urls = {
    'train.mat': 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
    'test.mat': 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
    'extra.mat': 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'
}

data_urls = [
    mnist_urls,
    svhn_urls
]
data_folder = [
    'mnist',
    'svhn'
]

if not op.exists(folder):
    os.mkdir(folder)
    for i in range(len(data_urls)):
        if not op.exists(folder + data_folder[i]):
            os.mkdir(folder + data_folder[i])
            print('Downloading data for ' + data_folder[i])
            for filename, url in data_urls[i].items():
                urlretrieve(url, folder + data_folder[i] + '/' + filename)

