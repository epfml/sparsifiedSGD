import argparse
import os
import pickle

import numpy as np

from logistic_parallel import LogisticParallelSGD
from parameters import Parameters
from utils import pickle_it

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('result_dir', type=str)
args = parser.parse_args()

DATA_DIR = args.data_dir
RESULT_DIR = args.result_dir
DATASET = 'rcv1.pickle'

print('load dataset')
dataset = os.path.join(DATA_DIR, DATASET)
with open(dataset, 'rb') as f:
    X, y = pickle.load(f)

n, d = X.shape


def run_parallel_experiment(directory, dataset_pickle, models, cores, baseline, repeat=3):
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle_it([m(1) for m in models], 'models', directory)
    pickle_it(cores, 'cores', directory)

    print('load dataset')
    with open(dataset_pickle, 'rb') as f:
        X, y = pickle.load(f)

    print('start experiment')

    chronos = np.zeros((len(models), len(cores), repeat))
    stop_times = np.zeros((len(models), len(cores), repeat), dtype=int)

    for r in range(repeat):
        for c_idx, core in enumerate(cores):
            for m_idx, model in enumerate(models):
                p = model(core)
                print("{} - cores {} - repeat {}".format(p, core, r))
                m = LogisticParallelSGD(p)
                timing, epoch, iteration, losses = m.fit_until(X, y, num_features=X.shape[1], num_samples=X.shape[0],
                                                               baseline=baseline)
                chronos[m_idx, c_idx, r] = timing
                stop_times[m_idx, c_idx, r] = epoch * X.shape[0] + iteration

                pickle_it(chronos, 'chronos', directory)
                pickle_it(stop_times, 'stop_times', directory)

    pickle_it(chronos, 'chronos', directory)
    pickle_it(stop_times, 'stop_times', directory)
    print('results saved in "{}"'.format(directory))


models = [
    lambda n_cores: Parameters(name="top100", num_epoch=6, lr_type='decay', initial_lr=2., n_cores=n_cores,
                               tau=10 / 100 * d,
                               regularizer=1 / n, estimate='final', take_k=100, take_top=True,
                               with_memory=True),
    lambda n_cores: Parameters(name="rand100", num_epoch=6, lr_type='decay', initial_lr=2., n_cores=n_cores,
                               tau=10 / 100 * d,
                               regularizer=1 / n, estimate='final', take_k=100, take_top=False,
                               with_memory=True),
    lambda n_cores: Parameters(name="hogwild", num_epoch=6, lr_type='decay', initial_lr=2., n_cores=n_cores,
                               tau=10, regularizer=1 / n,
                               estimate='final'),
]

cores = [1, 2, 3, 5, 8, 10, 12, 14, 16, 18, 20, 22, 24]
baseline = 0.09

run_parallel_experiment(args.directory, dataset, models, cores, baseline, repeat=3)
