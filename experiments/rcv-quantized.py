import argparse
import multiprocessing as mp
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from logistic import LogisticSGD
from parameters import Parameters
from utils import pickle_it

plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('result_dir', type=str)
args = parser.parse_args()

DATA_DIR = args.data_dir
RESULT_DIR = args.result_dir
DATASET = 'rcv1.pickle'
NUM_EPOCH = 10

print('load dataset')
dataset = os.path.join(DATA_DIR, DATASET)
with open(dataset, 'rb') as f:
    X, y = pickle.load(f)
n, d = X.shape

params = []

params.append(Parameters(name="full-sgd", num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=10.,
                         regularizer=1 / n, estimate='mean'))
params.append(Parameters(name="top1", num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=10.,
                         regularizer=1 / n, estimate='mean',
                         take_k=1, take_top=True, with_memory=True))
params.append(Parameters(name="rand1", num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=1.,
                         regularizer=1 / n, estimate='mean',
                         take_k=1, take_top=False, with_memory=True))
params.append(Parameters(name="rand10", num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=1.,
                         regularizer=1 / n, estimate='mean',
                         take_k=10, take_top=False, with_memory=True))
params.append(Parameters(name="qsgd-8bit", num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=10.,
                         regularizer=1 / n, estimate='mean',
                         qsgd_s=2 ** 8))
params.append(Parameters(name="qsgd-4bit", num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=10.,
                         regularizer=1 / n, estimate='mean',
                         qsgd_s=2 ** 4))
params.append(Parameters(name="qsgd-2bit", num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=10.,
                         regularizer=1 / n, estimate='mean',
                         qsgd_s=2 ** 2))


def run_logistic(param):
    m = LogisticSGD(param)
    res = m.fit(X, y)
    print('{} - score: {}'.format(param, m.score(X, y)))
    return res


if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
pickle_it(params, 'params', RESULT_DIR)

print('start experiment')
with mp.Pool(len(params)) as pool:
    results = pool.map(run_logistic, params)

pickle_it(results, 'results', RESULT_DIR)
print('results saved in "{}"'.format(RESULT_DIR))
