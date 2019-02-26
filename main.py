from model.model import CustomModel
from model.preprocessing import get_data

BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 10
SAVE_DIR = 'saved_models/'
MODEL_NAME = 'cifar10-cnn.h5'
SUBSET = None

(x_train, y_train), (x_test, y_test) = get_data(num_classes=NUM_CLASSES, subset=SUBSET)
model = CustomModel()
model._build(num_classes=NUM_CLASSES)
model._compile()
model._fit(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE, epochs=EPOCHS)
model._save(save_dir=SAVE_DIR, model_name=MODEL_NAME)
model._evaluate(x_test, y_test)