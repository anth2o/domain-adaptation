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
        domain_predictions = self._build_domain_classifier(features, num_domains)
        self.model = Model(inputs=inputs, outputs=label_predictions)
        self.model_unlabelled = Model(inputs=inputs, outputs=domain_predictions)

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

        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)

        features = Flatten()(x)
        return inputs, features

    def _build_label_predictor(self, features, num_classes):
        x = Dense(32, activation='relu')(features)
        outputs = Dense(num_classes, activation='softmax', name='label_predictor')(x)
        return outputs

    def _build_domain_classifier(self, features, num_domains):
        x = GRL()(features)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(num_domains, activation='softmax', name='domain_classifier')(x)
        return outputs

    def _compile(self):
        if not self.model and not self.model_unlabelled:
            raise Exception("Trying to compile model but it isn't built")
        opt = rmsprop(lr=10e-4, decay=1e-6)
        loss={'label_predictor': 'categorical_crossentropy'}
        self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
        loss={'domain_classifier': 'categorical_crossentropy'}
        self.model_unlabelled.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    def _fit(self, x_train, y_train, x_test, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=LOG_FILE):
        if not self.model:
            raise Exception("Trying to fit model but it isn't built")
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=10e-4, patience=10, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=10e-7, verbose=1)
        csv_logger = CSVLogger(log_file)
        for i in range(epochs):
            self.model_unlabelled.fit(x_train_unlabelled, y_train_unlabelled['domain'],
                batch_size=batch_size,
                epochs=1,
                validation_data=(x_test_unlabelled, y_test_unlabelled['domain']),
                shuffle=True,)
            self.model.fit(x_train, y_train['label'],
                batch_size=batch_size,
                epochs=1,
                validation_data=(x_test, y_test['label']),
                shuffle=True,)

    def _run_all(self, x_train, x_test, y_train, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=LOG_FILE, save_dir=SAVE_DIR, model_name=MODEL_NAME):
        self._build(num_classes=num_classes)
        self._compile()
        print(self.model.summary())
        self._fit(x_train, y_train, x_test, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled, batch_size=batch_size, epochs=epochs, log_file=log_file)
        self._save(save_dir=save_dir, model_name=model_name)
        self._evaluate(x_test, y_test['label'])




