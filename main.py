import numpy as np

from models.cnn import CNN
from models.cnn_grl import CNNGRL
from utils.preprocessing import Preprocessor
from utils.config import *

pp = Preprocessor(num_classes=NUM_CLASSES, domains=DOMAINS, ignore_labels=IGNORE_LABELS, image_size=IMAGE_SIZE, channels=CHANNELS, subset=SUBSET)
((x_train, y_train), (x_test, y_test)), ((x_train_unlabelled, y_train_unlabelled), (x_test_unlabelled, y_test_unlabelled)) = pp.get_data()

architecture = 'CNNGRL'

if architecture == 'CNN':
    model = CNN()
    y_train = y_train[0]
    y_test = y_test[0]
elif architecture == 'CNNGRL':
    model = CNNGRL()

model._run_all(x_train, x_test, y_train, y_test, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=LOG_FILE, save_dir=SAVE_DIR, model_name=MODEL_NAME)
