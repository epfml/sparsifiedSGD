import argparse
import multiprocessing as mp
import os
import pickle

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
DATASET = 'epsilon.pickle'
NUM_EPOCH = 3

print('load dataset')
dataset = os.path.join(DATA_DIR, DATASET)
with open(dataset, 'rb') as f:
    X, y = pickle.load(f)

n, d = X.shape

params = []

params.append(Parameters(name="full-sgd", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=d,
                         regularizer=1 / n, estimate='(t+tau)^2'))
params.append(Parameters(name="full-sgd-no-shift", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=1,
                         regularizer=1 / n, estimate='(t+tau)^2'))
params.append(Parameters(name="top1-no-memory", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=d,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=False, take_top=True))
params.append(Parameters(name="top1", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=d,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True, take_top=True))
params.append(Parameters(name="top1-no-shift", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=1,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True, take_top=True))
params.append(Parameters(name="rand1-d", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=d,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True, take_top=False))
params.append(Parameters(name="rand1-no-shift", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=1,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True))
params.append(Parameters(name="rand2-d", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=d,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=2, with_memory=True))
params.append(Parameters(name="rand2-d/k", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=d/2,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=2, with_memory=True))
params.append(Parameters(name="rand2-no-shift", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=1,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=2, with_memory=True))
params.append(Parameters(name="rand3-d", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=d,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=3, with_memory=True))
params.append(Parameters(name="rand3-d/k", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=d/3,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=3, with_memory=True))
params.append(Parameters(name="rand3-no-shift", num_epoch=NUM_EPOCH, lr_type='decay', initial_lr=2, tau=1,
                         regularizer=1 / n, estimate='(t+tau)^2', take_k=3, with_memory=True))


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

# res_and_infos = []
# names = []
# lrs = []
# for p, res in zip(params, results):
#     lr = p.initial_lr
#     name = str(p)[:-(len(str(lr)) + 1)]
#     names.append(name)
#     lrs.append(lr)
#     res_and_infos.append((name, lr, res[1][:-1]))
#
# names = sorted(list(set(names)))
# lrs = sorted(list(set(lrs)))
#
# # plot
# f, axarr = plt.subplots(1, len(names), figsize=(20, 4), sharey=True)
#
# for name, lr, loss in res_and_infos:
#     ax = axarr[names.index(name)]
#     idx = lrs.index(lr)
#     ax.plot(loss, "C{}".format(idx), label=str(lr))
#
# for name, ax in zip(names, axarr):
#     ax.set_title(name)
#     ax.set_ylim(top=2.)
#
# axarr[0].set_ylabel('loss')
# axarr[0].legend();
# result_pdf = os.path.join(RESULT_DIR, 'figure.pdf')
# f.savefig(result_pdf)
# print('figure saved in {}'.format(result_pdf))
