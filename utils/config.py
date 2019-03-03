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

EPOCHS = 500
SUBSET = 10000

CNN_GRL_MODEL_NAME = 'cnn_grl_merge_train_svhn_2'
CNN_GRL_LOG_FILE = 'logs/' + CNN_GRL_MODEL_NAME + '.log'
CNN_MODEL_NAME = 'cnn_train_svhn'
CNN_LOG_FILE = 'logs/' + CNN_GRL_MODEL_NAME + '.log'

if CONFIG == 'DEBUG':
    EPOCHS = 40
    SUBSET = 100
    MODEL_NAME = 'debug'
    LOG_FILE = 'logs/' + MODEL_NAME + '.log'
    CNN_GRL_MODEL_NAME = MODEL_NAME
    CNN_GRL_LOG_FILE = LOG_FILE
    CNN_MODEL_NAME = MODEL_NAME
    CNN_LOG_FILE = LOG_FILE
