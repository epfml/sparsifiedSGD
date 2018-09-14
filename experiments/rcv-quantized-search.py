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
from utils import pickle_it, unpickle_dir

plt.switch_backend('agg')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('result_dir', type=str)
args = parser.parse_args()

DATA_DIR = args.data_dir
RESULT_DIR = args.result_dir
DATASET = 'rcv1.pickle'
SUBSAMPLE = 0.01
SEED = 2018
NUM_EPOCH = 5

print('load dataset')
dataset = os.path.join(DATA_DIR, DATASET)
with open(dataset, 'rb') as f:
    X, y = pickle.load(f)

print('down sample dataset')
np.random.seed(SEED)
n, d = X.shape
sub_idx = np.random.choice(n, int(SUBSAMPLE * n), replace=False)
X, y = X[sub_idx, :], y[sub_idx]

params = []
lrs = [0.1, 1., 10., 100.]

for lr in lrs:
    params.append(Parameters(name="full-sgd-{}".format(lr), num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=lr,
                             regularizer=1 / n, estimate='mean'))
    params.append(Parameters(name="top1-{}".format(lr), num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=lr,
                             regularizer=1 / n, estimate='mean',
                             take_k=1, take_top=True, with_memory=True))
    params.append(Parameters(name="rand1-{}".format(lr), num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=lr,
                             regularizer=1 / n, estimate='mean',
                             take_k=1, take_top=False, with_memory=True))
    params.append(Parameters(name="rand10-{}".format(lr), num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=lr,
                             regularizer=1 / n, estimate='mean',
                             take_k=10, take_top=False, with_memory=True))
    params.append(Parameters(name="qsgd-8bit-{}".format(lr), num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=lr,
                             regularizer=1 / n, estimate='mean',
                             qsgd_s=2 ** 8))
    params.append(Parameters(name="qsgd-4bit-{}".format(lr), num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=lr,
                             regularizer=1 / n, estimate='mean',
                             qsgd_s=2 ** 4))
    params.append(Parameters(name="qsgd-2bit-{}".format(lr), num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=lr,
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
