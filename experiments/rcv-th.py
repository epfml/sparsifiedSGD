import argparse
import multiprocessing as mp
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import matplotlib.pyplot as plt

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
NUM_EPOCH = 5

print('load dataset')
dataset = os.path.join(DATA_DIR, DATASET)
with open(dataset, 'rb') as f:
    X, y = pickle.load(f)
n, d = X.shape

params = []

params.append(Parameters(name="full-sgd", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10*d,
                         regularizer=1 / n, estimate='(t+tau)^2'))
params.append(Parameters(name="full-sgd-no-shift", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10,
                         regularizer=1 / n, estimate='(t+tau)^2'))
params.append(Parameters(name="top1", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10 * d,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True, take_top=True))
params.append(Parameters(name="top1-no-shift", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True, take_top=True))
params.append(Parameters(name="top10", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10 * d,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=10, with_memory=True, take_top=True))
params.append(Parameters(name="top10-d/k", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10 * d / 10,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=10, with_memory=True, take_top=True))
params.append(Parameters(name="top10-no-shift", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=10, with_memory=True, take_top=True))
params.append(Parameters(name="rand10", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10 * d,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=10, with_memory=True))
params.append(Parameters(name="rand10-d/k", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10 * d / 10,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=10, with_memory=True))
params.append(Parameters(name="rand10-no-shift", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=10, with_memory=True))
params.append(Parameters(name="rand20", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10 * d,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=20, with_memory=True))
params.append(Parameters(name="rand20-d/k", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10 * d / 20,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=20, with_memory=True))
params.append(Parameters(name="rand20-no-shift", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=20, with_memory=True))
params.append(Parameters(name="rand30", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10 * d,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=30, with_memory=True))
params.append(Parameters(name="rand30-d/k", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10 * d / 30,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=30, with_memory=True))
params.append(Parameters(name="rand30-no-shift", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=10,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=30, with_memory=True))


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

