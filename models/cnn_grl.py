from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Concatenate
from keras.models import Model
from keras.optimizers import rmsprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
import numpy as np

from .base_model import BaseModel
from layers.grl import GRL
from utils.config import *

class CNNGRL(BaseModel):     
    def __init__(self, image_size=IMAGE_SIZE, channels=CHANNELS):
        super(CNNGRL, self).__init__()
        self.loss = {'domain_classifier': 'categorical_crossentropy', 'label_predictor': 'categorical_crossentropy'}
        self.input_shape = image_size + (channels,)

    def _build(self, num_classes=NUM_CLASSES, num_domains=len(DOMAINS)):
        inputs, features = self._build_feature_extractor()
        feature_extractor = Model(inputs=inputs, outputs=features)
        inputs_label, label_predictions = self._build_label_predictor(feature_extractor, num_classes)
        inputs_domain, domain_predictions = self._build_domain_classifier(feature_extractor, num_domains)
        self.model_label = Model(inputs=inputs_label, outputs=label_predictions)
        self.model = Model(inputs=[inputs_label, inputs_domain], outputs=[label_predictions, domain_predictions])

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
        x = Dense(512, activation='relu')(x)
        features = Dropout(0.5)(x)

        return inputs, features

    def _build_label_predictor(self, feature_extractor, num_classes):
        inputs_label = Input(shape=self.input_shape)
        x = feature_extractor(inputs_label)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax',
                        name='label_predictor')(x)
        return inputs_label, outputs

    def _build_domain_classifier(self, feature_extractor, num_domains):
        inputs_domain = Input(shape=self.input_shape)
        x = feature_extractor(inputs_domain)
        x = GRL(0.01)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(num_domains, activation='softmax',
                        name='domain_classifier')(x)
        return inputs_domain, outputs

    def _compile(self):
        if not self.model:
            raise Exception("Trying to compile model but it isn't built")
        self.model_label.compile(loss=self.loss['label_predictor'], optimizer=self.opt, metrics=['accuracy'])
        self.model.compile(loss=self.loss, optimizer=self.opt, metrics=['accuracy'])

    def _fit(self, x_train, y_train, x_test, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=CNN_GRL_LOG_FILE):
        if not self.model:
            raise Exception("Trying to fit model but it isn't built")
        # early_stopping = EarlyStopping(monitor='val_loss', min_delta=10e-4, patience=10, restore_best_weights=True, verbose=1)
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=10e-7, verbose=1)
        # csv_logger = CSVLogger(log_file)
        x_train_unlabelled = np.concatenate([x_train_unlabelled, x_train], axis=0)
        y_train_unlabelled = np.concatenate([y_train_unlabelled['domain'], y_train['domain']], axis=0)
        assert x_train_unlabelled.shape[0] == y_train_unlabelled.shape[0]
        p = np.random.permutation(x_train_unlabelled.shape[0])
        x_train_unlabelled = x_train_unlabelled[p][:x_train.shape[0]]
        y_train_unlabelled = y_train_unlabelled[p][:y_train['domain'].shape[0]]
        self.model.fit([x_train, x_train_unlabelled],
            [y_train['label'], y_train_unlabelled],
            batch_size=batch_size,
            epochs=epochs,
            # validation_data=(
            #     [x_test, np.concatenate([x_test_unlabelled, x_test], axis=0)],
            #     [y_test['label'], np.concatenate([y_test_unlabelled['domain'], y_test['domain']], axis=0)]
            #     ),
            shuffle=True
            )
        print(self.model.weights[1:4] == self.model_label.weights[1:4])

    def _evaluate(self, x_test, y_test):
        if not self.model:
            raise Exception("Trying to evaluate model but it isn't built")
        scores = self.model_label.evaluate(x_test, y_test, verbose=0)
        print('Test accuracy:', scores[1])

    def _run_all(self, x_train, x_test, y_train, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=CNN_GRL_LOG_FILE, save_dir=SAVE_DIR, model_name=CNN_GRL_MODEL_NAME):
        self._build(num_classes=num_classes)
        self._compile()
        print(self.model.summary())
        self._fit(x_train, y_train, x_test, y_test, x_train_unlabelled, y_train_unlabelled,
                  x_test_unlabelled, y_test_unlabelled, batch_size=batch_size, epochs=epochs, log_file=log_file)
        self._save(save_dir=save_dir, model_name=model_name)
        self._evaluate(x_test, y_test['label'])       
