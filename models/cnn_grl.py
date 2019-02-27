from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.models import Model
from keras.optimizers import rmsprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
import numpy as np

from .base_model import BaseModel
from layers.grl import GRL
from utils.config import *


class CNNGRL(BaseModel):
    def __init__(self):
        self.model = None
        self.model_unlabelled = None

    def _build(self, num_classes=NUM_CLASSES, num_domains=len(DOMAINS)):
        inputs, features = self._build_feature_extractor()
        label_predictions = self._build_label_predictor(features, num_classes)
        domain_predictions = self._build_domain_classifier(
            features, num_domains)
        self.model = Model(inputs=inputs, outputs=label_predictions)
        self.model_unlabelled = Model(
            inputs=inputs, outputs=domain_predictions)

    def _build_feature_extractor(self):
        inputs = Input(shape=(32, 32, 3))
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        features = Dropout(0.5)(x)

        return inputs, features

    def _build_label_predictor(self, features, num_classes):
        x = Dense(32, activation='relu')(features)
        outputs = Dense(num_classes, activation='softmax',
                        name='label_predictor')(x)
        return outputs

    def _build_domain_classifier(self, features, num_domains):
        x = GRL(0.01)(features)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(num_domains, activation='softmax',
                        name='domain_classifier')(x)
        return outputs

    def _compile(self):
        if not self.model and not self.model_unlabelled:
            raise Exception("Trying to compile model but it isn't built")
        self.model.compile(loss=self.loss, optimizer=self.opt, metrics=['accuracy'])
        self.model_unlabelled.compile(loss=self.loss, optimizer=self.opt, metrics=['accuracy'])

    def _fit(self, x_train, y_train, x_test, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=CNN_GRL_LOG_FILE):
        if not self.model:
            raise Exception("Trying to fit model but it isn't built")
        for _ in range(epochs):
            self.model_unlabelled.fit(np.concatenate([x_train_unlabelled, x_train], axis=0), np.concatenate([y_train_unlabelled['domain'], y_train['domain']], axis=0),
                                      batch_size=batch_size,
                                      epochs=1,
                                      validation_data=(
                                          np.concatenate([x_test_unlabelled, x_test], axis=0), np.concatenate([y_test_unlabelled['domain'], y_test['domain']])),
                                      shuffle=True,)
            self.model.fit(x_train, y_train['label'],
                           batch_size=batch_size,
                           epochs=1,
                           validation_data=(x_test, y_test['label']),
                           shuffle=True,)

    def _run_all(self, x_train, x_test, y_train, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=CNN_GRL_LOG_FILE, save_dir=SAVE_DIR, model_name=CNN_GRL_MODEL_NAME):
        self._build(num_classes=num_classes)
        self._compile()
        print(self.model.summary())
        print(self.model_unlabelled.summary())
        self._fit(x_train, y_train, x_test, y_test, x_train_unlabelled, y_train_unlabelled,
                  x_test_unlabelled, y_test_unlabelled, batch_size=batch_size, epochs=epochs, log_file=log_file)
        self._save(save_dir=save_dir, model_name=model_name)
        self._evaluate(x_test, y_test['label'])
