CONFIG = 'PROD'

BATCH_SIZE = 32
NUM_CLASSES = 10
SAVE_DIR = 'weights/'
IMAGE_SIZE = (32, 32)
CHANNELS = 3
DOMAINS = [
    'mnist',
    'svhn'
]
DOMAINS_IGNORE_LABELS = [
    'mnist'
]

EPOCHS = 200
SUBSET = None
MODEL_NAME = 'cnn_train_svhn.h5'
LOG_FILE = 'logs/cnn_train_svhn.log'

if CONFIG == 'DEBUG':
    EPOCHS = 10
    SUBSET = 10
    MODEL_NAME = 'debug.h5'
    LOG_FILE = 'logs/debug.log'
