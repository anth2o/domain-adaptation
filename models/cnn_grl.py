from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Concatenate
from keras.models import Model
from keras.optimizers import rmsprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
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
        self.feature_extractor = Model(inputs=inputs, outputs=features)
        inputs_label, label_predictions = self._build_label_predictor(num_classes)
        inputs_domain, input_lambda, domain_predictions = self._build_domain_classifier(num_domains)
        self.model_label = Model(inputs=inputs_label, outputs=label_predictions)
        self.model = Model(inputs=[inputs_label, inputs_domain, input_lambda], outputs=[label_predictions, domain_predictions])

    def _build_feature_extractor(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        features = Dense(512, activation='relu', name='dense_1')(x)

        return inputs, features

    def _build_label_predictor(self, num_classes):
        inputs_label = Input(shape=self.input_shape)
        x = self.feature_extractor(inputs_label)
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu', name='dense_2')(x)
        outputs = Dense(num_classes, activation='softmax',
                        name='label_predictor')(x)
        return inputs_label, outputs

    def _build_domain_classifier(self, num_domains):
        inputs_domain = Input(shape=self.input_shape)
        input_lambda = Input(shape=(1,))
        x = self.feature_extractor(inputs_domain)
        x = GRL()([x, input_lambda])
        x = Dense(512, activation='relu', name='dense_4')(x)
        x = Dense(32, activation='relu', name='dense_5')(x)
        outputs = Dense(num_domains, activation='softmax',
                        name='domain_classifier')(x)
        return inputs_domain, input_lambda, outputs

    def _compile(self):
        if not self.model:
            raise Exception("Trying to compile model but it isn't built")
        self.model_label.compile(loss=self.loss['label_predictor'], optimizer=self.opt, metrics=['accuracy'])
        self.model.compile(loss=self.loss, optimizer=self.opt, metrics=['accuracy'], loss_weights=self.loss_weights)

    def _fit(self, x_train, y_train, x_test, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=CNN_GRL_LOG_FILE):
        if not self.model:
            raise Exception("Trying to fit model but it isn't built")
        reduce_lr_label = ReduceLROnPlateau(monitor='val_label_predictor_loss', factor=0.2, patience=20, min_lr=10e-8, verbose=1)
        reduce_lr_domain = ReduceLROnPlateau(monitor='val_domain_classifier_loss', factor=0.2, patience=20, min_lr=10e-8, verbose=1)
        csv_logger = CSVLogger(log_file)
        train_datagen = Generator(x_train, y_train, x_train_unlabelled, y_train_unlabelled, batch_size=batch_size, max_epochs=epochs, print_lambda=True)
        test_datagen = Generator(x_test, y_test, x_test_unlabelled, y_test_unlabelled, batch_size=batch_size, max_epochs=epochs, print_lambda=False)
        self.model.fit_generator(train_datagen,
            epochs=epochs,
            shuffle=False,
            validation_data=test_datagen,
            callbacks=[csv_logger]
            )

    def _evaluate(self, x_test, y_test):
        if not self.model:
            raise Exception("Trying to evaluate model but it isn't built")
        scores = self.model_label.evaluate(x_test, y_test, verbose=0)
        print('Test accuracy:', scores[1])

    def _run_all(self, x_train, x_test, y_train, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled, model_path=None, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=CNN_GRL_LOG_FILE, save_dir=SAVE_DIR, model_name=CNN_GRL_MODEL_NAME):
        self._build(num_classes=num_classes)
        if model_path is not None:
            if os.path.isfile(model_path):
                print('Loading weights from: ' + model_path)
                self.feature_extractor.load_weights(model_path, by_name=True)
                for layer in self.feature_extractor.layers[2:-2]:
                    print('Freeze layer: ' + str(layer.name))
                    layer.trainable=False
        print(self.model.summary())
        print(self.feature_extractor.summary())
        self._compile()
        self._fit(x_train, y_train, x_test, y_test, x_train_unlabelled, y_train_unlabelled,
                  x_test_unlabelled, y_test_unlabelled, batch_size=batch_size, epochs=epochs, log_file=log_file)
        self._save(save_dir=save_dir, model_name=model_name)
        self._evaluate(x_test, y_test['label'])       
