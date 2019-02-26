import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.models import Model
import os


class CustomModel:
    def _build(self, num_classes):
    #     model.add(Lambda(augment_2d,
    #                     input_shape=x_train.shape[1:],
    #                     arguments={'rotation': 8.0, 'horizontal_flip': True}))
        inputs = Input(shape=(32, 32, 3))
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=predictions)

    def _compile(self):
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

    def _fit(self, x_train, y_train, x_test, y_test, batch_size=32, epochs=5):
        self.model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

    def _save(self, save_dir, model_name):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    def _evaluate(self, x_test, y_test):
        scores = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
