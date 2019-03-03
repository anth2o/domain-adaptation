from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Concatenate
from keras.models import Model
from keras.optimizers import rmsprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.utils import plot_model
import numpy as np
import os

from .base_model import BaseModel
from layers.grl import GRL
from utils.config import *
from utils.generator import Generator

class CNNGRL(BaseModel):     
    def __init__(self, image_size=IMAGE_SIZE, channels=CHANNELS):
        super(CNNGRL, self).__init__()
        self.loss = {'domain_classifier': 'categorical_crossentropy', 'label_predictor': 'categorical_crossentropy'}
        self.loss_weights = {'domain_classifier': 1, 'label_predictor': 1}
        self.input_shape = image_size + (channels,)

    def _build(self, num_classes=NUM_CLASSES, num_domains=len(DOMAINS)):
        inputs, features = self._build_feature_extractor()
        self.feature_extractor = Model(inputs=inputs, outputs=features, name='feature_extractor')
        inputs_label, label_predictions = self._build_label_predictor(num_classes)
        inputs_domain, input_lambda, domain_predictions = self._build_domain_classifier(num_domains)
        self.model_label = Model(inputs=inputs_label, outputs=label_predictions)
        self.model = Model(inputs=[inputs_label, inputs_domain, input_lambda], outputs=[label_predictions, domain_predictions])

    def _build_feature_extractor(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv2d_1')(inputs)
        x = Conv2D(32, (3, 3), activation='relu', name='conv2d_2')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1')(x)
        x = Dropout(0.25, name='dropout_1')(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2d_3')(x)
        x = Conv2D(64, (3, 3), activation='relu', name='conv2d_4')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2')(x)
        x = Dropout(0.25, name='dropout_2')(x)

        x = Flatten(name='flatten_1')(x)
        features = Dense(512, activation='relu', name='dense_1')(x)

        return inputs, features

    def _build_label_predictor(self, num_classes):
        inputs_label = Input(shape=self.input_shape, name='input_labels')
        x = self.feature_extractor(inputs_label)
        x = Dropout(0.5, name='dropout_label_1')(x)
        x = Dense(32, activation='relu', name='dense_label_1')(x)
        outputs = Dense(num_classes, activation='softmax',
                        name='label_predictor')(x)
        return inputs_label, outputs

    def _build_domain_classifier(self, num_domains):
        inputs_domain = Input(shape=self.input_shape, name='input_domains')
        input_lambda = Input(shape=(1,), name='input_lambda')
        x = self.feature_extractor(inputs_domain)
        x = GRL()([x, input_lambda])
        x = Dense(512, activation='relu', name='dense_domain_1')(x)
        x = Dense(32, activation='relu', name='dense_domain_2')(x)
        outputs = Dense(num_domains, activation='softmax',
                        name='domain_classifier')(x)
        return inputs_domain, input_lambda, outputs

    def _compile(self):
        if not self.model:
            raise Exception("Trying to compile model but it isn't built")
        self.model_label.compile(loss=self.loss['label_predictor'], optimizer=self.opt, metrics=['accuracy'])
        self.model.compile(loss=self.loss, optimizer=self.opt, metrics=['accuracy'], loss_weights=self.loss_weights)

    def _fit(self, x_train, y_train, x_test, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=None):
        if not self.model:
            raise Exception("Trying to fit model but it isn't built")
        reduce_lr_label = ReduceLROnPlateau(monitor='val_label_predictor_loss', factor=0.2, patience=20, min_lr=10e-8, verbose=1)
        reduce_lr_domain = ReduceLROnPlateau(monitor='val_domain_classifier_loss', factor=0.2, patience=20, min_lr=10e-8, verbose=1)
        csv_logger = CSVLogger(log_file)
        train_datagen = Generator(x_train, y_train, x_train_unlabelled, y_train_unlabelled, batch_size=batch_size, max_epochs=epochs, print_lambda=False)
        test_datagen = Generator(x_test, y_test, x_test_unlabelled, y_test_unlabelled, batch_size=batch_size, max_epochs=epochs, print_lambda=False)
        self.model.fit_generator(train_datagen,
            epochs=epochs,
            shuffle=False,
            validation_data=test_datagen,
            callbacks=[csv_logger]
            )

    def _save(self, model_name, save_dir=SAVE_DIR):
        if not self.model:
            raise Exception("Trying to save model but it isn't built")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        self._unfreeze_layers()
        self.feature_extractor.save(model_path + '/feature_extractor.h5')
        self.model.save(model_path + '/model.h5')
        print('Saved trained model at %s ' % model_path)
        self._freeze_layers()

    def _evaluate(self, x_test, y_test):
        if not self.model:
            raise Exception("Trying to evaluate model but it isn't built")
        scores = self.model_label.evaluate(x_test, y_test, verbose=0)
        print('Test accuracy:', scores[1])

    def _run_all(self, x_train, x_test, y_train, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled, model_name, pre_trained_model_name, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, epochs=EPOCHS, save_dir=SAVE_DIR):
        log_file = 'logs/' + model_name + '.log'
        self._build(num_classes=num_classes)
        self._load_pre_trained_weights(pre_trained_model_name)
        print(self.model.summary())
        print(self.feature_extractor.summary())
        self._compile()
        self._fit(x_train, y_train, x_test, y_test, x_train_unlabelled, y_train_unlabelled,
                  x_test_unlabelled, y_test_unlabelled, batch_size=batch_size, epochs=epochs, log_file=log_file)
        self._save(save_dir=save_dir, model_name=model_name)
        self._evaluate(x_test, y_test['label'])    

    def _load_pre_trained_weights(self, pre_trained_model_name):
        pre_trained_model_path = 'weights/' + pre_trained_model_name + '.h5'
        if pre_trained_model_path is not None:
            if os.path.isfile(pre_trained_model_path):
                print('Loading weights from: ' + pre_trained_model_path)
                self.feature_extractor.load_weights(pre_trained_model_path, by_name=True)
                self._freeze_layers()

    def _load_weights(self, model_name):
        self._unfreeze_layers()
        self.feature_extractor.load_weights('weights/' + model_name + '/feature_extractor.h5', by_name=True)
        self.model.load_weights('weights/' + model_name + '/model.h5', by_name=True)
        self._freeze_layers()

    def _freeze_layers(self, verbose=False):
        for layer in self.feature_extractor.layers[2:-2]:
            if verbose:
                print('Freeze layer: ' + str(layer.name))
            layer.trainable=False

    def _unfreeze_layers(self):
        for layer in self.feature_extractor.layers:
            layer.trainable = True

    def _plot_model(self):
        model_name = 'cnn_grl'
        model_path = 'img/' + model_name
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        plot_model(self.model, to_file=model_path + '/model.png', show_layer_names=True, show_shapes=True)
        plot_model(self.feature_extractor, to_file=model_path + '/feature_extractor.png', show_layer_names=True, show_shapes=True)

