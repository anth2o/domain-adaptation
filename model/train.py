import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.models import Model
import os

from preprocessing import get_data

BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 10
SAVE_DIR = 'saved_models/'
MODEL_NAME = 'base.h5'

def build_model():
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
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    return model

def compile_model(model):
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    return model



(x_train, y_train), (x_test, y_test) = get_data(NUM_CLASSES)
model = build_model()
model = compile_model(model)

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_test, y_test),
          shuffle=True)

# Save model and weights
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
model_path = os.path.join(SAVE_DIR, MODEL_NAME)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])