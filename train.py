import numpy as np

from models.cnn import CNN
from models.cnn_grl import CNNGRL
from utils.preprocessing import Preprocessor
from utils.config import *
import argparse

parser = argparse.ArgumentParser(description='Train models')
parser.add_argument('--model', type=str, help='the architecure: cnn or cnn_grl', default=None)
parser.add_argument('--source', type=str, nargs='+', help='the source data: svhn or mnist', default=['svhn'])

args = parser.parse_args()

domains_ignore_labels = list(set(DOMAINS) - set(args.source))

pp = Preprocessor(domains_ignore_labels=domains_ignore_labels)
((x_train, y_train), (x_test, y_test)), ((x_train_unlabelled, y_train_unlabelled), (x_test_unlabelled, y_test_unlabelled)) = pp.get_data()

if CONFIG == 'PROD':    
    model_name = args.model + '_train_' + '_'.join(args.source)
elif CONFIG == 'DEBUG':
    model_name = 'debug'

print(model_name)

if args.model == 'cnn':
    model = CNN()
    model._run_all(x_train, x_test, y_train['label'], y_test['label'], model_name=model_name)
elif args.model == 'cnn_grl':
    pretrained_model_name = 'cnn_train_' + '_'.join(args.source)
    print(pretrained_model_name)
    model = CNNGRL()
    model._run_all(x_train, x_test, y_train, y_test, x_train_unlabelled, y_train_unlabelled, x_test_unlabelled, y_test_unlabelled, model_name=model_name, pre_trained_model_name=pretrained_model_name)
else:
    raise Exception('Invalid model choice')
