import keras
import numpy as np

def get_data(num_classes):
    (x_train, y_train), (x_test, y_test) = (np.ones((1, 32, 32, 3)), np.ones((1, 1))), (np.ones((1, 32, 32, 3)), np.ones((1, 1)))
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
