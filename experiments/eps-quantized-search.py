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
DATASET = 'epsilon.pickle'
SUBSAMPLE = 0.01
SEED = 2018
NUM_EPOCH = 10

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
lrs = [0.01, 0.1, 1., 10., 100.]

for lr in lrs:
    params.append(Parameters(name="full-sgd-{}".format(lr), num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=lr,
                             regularizer=1 / n, estimate='mean'))
    params.append(Parameters(name="top1-{}".format(lr), num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=lr,
                             regularizer=1 / n, estimate='mean',
                             take_k=1, take_top=True, with_memory=True))
    params.append(Parameters(name="rand1-{}".format(lr), num_epoch=NUM_EPOCH, lr_type='bottou', initial_lr=lr,
                             regularizer=1 / n, estimate='mean',
                             take_k=1, take_top=False, with_memory=True))
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

# process data

res_and_infos = []
names = []
lrs = []
for p, res in zip(params, results):
    lr = p.initial_lr
    name = str(p)[:-(len(str(lr)) + 1)]
    names.append(name)
    lrs.append(lr)
    res_and_infos.append((name, lr, res[1][:-1]))

names = sorted(list(set(names)))
lrs = sorted(list(set(lrs)))

# plot
f, axarr = plt.subplots(1, len(names), figsize=(20, 4), sharey=True)

for name, ax in zip(names, axarr):
    ax.set_title(name)
    ax.set_xlabel('epoch')
    ax.set_ylim(0., 2.)

for name, lr, loss in res_and_infos:
    ax = axarr[names.index(name)]
    idx = lrs.index(lr)
    ax.plot(np.arange(len(loss)) / 10, loss, "C{}".format(idx), label=str(lr))


axarr[0].set_ylabel('loss')
axarr[0].legend();
result_pdf = os.path.join(RESULT_DIR, 'figure.pdf')
f.savefig(result_pdf)
print('figure saved in {}'.format(result_pdf))
