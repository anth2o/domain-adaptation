from model.model import CustomModel
from model.preprocessing import get_data

BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 10
SAVE_DIR = 'saved_models/'
MODEL_NAME = 'cnn_train_svhn.h5'
SUBSET = None
LOG_FILE = 'logs/cnn_train_svhn.log'

(x_train, y_train), (x_test, y_test) = get_data(num_classes=NUM_CLASSES, subset=SUBSET)
model = CustomModel()
model._build(num_classes=NUM_CLASSES)
model._compile()
model._fit(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=LOG_FILE)
model._save(save_dir=SAVE_DIR, model_name=MODEL_NAME)
model._evaluate(x_test, y_test)