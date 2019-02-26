import numpy as np

from models.cnn import CNN
from models.cnn_grl import CNNGRL
from utils.preprocessing import get_data
from utils.config import *


(x_train, y_train), (x_test, y_test) = get_data(num_classes=NUM_CLASSES, subset=SUBSET, num_domains=NUM_DOMAINS)

architecture = 'CNNGRL'

if architecture == 'CNN':
    model = CNN()
elif architecture == 'CNNGRL':
    model = CNNGRL()
    
model._run_all(x_train, x_test, y_train, y_test, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, epochs=EPOCHS, log_file=LOG_FILE, save_dir=SAVE_DIR, model_name=MODEL_NAME)
