from model.simple_cnn import SimpleCNN
from model.preprocessing import get_data
from utils.config import *

(x_train, y_train), (x_test, y_test) = get_data(num_classes=NUM_CLASSES, subset=SUBSET)
model = SimpleCNN()
model._build(num_classes=NUM_CLASSES)
model._compile()
model._fit(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=LOG_FILE)
model._save(save_dir=SAVE_DIR, model_name=MODEL_NAME)
model._evaluate(x_test, y_test)