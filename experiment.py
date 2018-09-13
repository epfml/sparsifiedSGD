import argparse
import multiprocessing as mp
import os
import pickle

import numpy as np

from logistic import LogisticSGD
from logistic_parallel import LogisticParallelSGD
from parameters import Parameters
from utils import pickle_it

X, y = None, None


def run_logistic(param):
    m = LogisticSGD(param)
    res = m.fit(X, y)
    print('{} - score: {}'.format(param, m.score(X, y)))
    return res


def run_experiment(directory, dataset_pickle, params, nproc=None):
    global X, y
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle_it(params, 'params', directory)

    print('load dataset')
    with open(dataset_pickle, 'rb') as f:
        X, y = pickle.load(f)

    print('start experiment')
    with mp.Pool(nproc) as pool:
        results = pool.map(run_logistic, params)

    pickle_it(results, 'results', directory)
    print('results saved in "{}"'.format(directory))


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('directory', type=str)
    parser.add_argument('--nproc', type=int, default=1)
    args = parser.parse_args()

    assert args.experiment in ['epsilon-th', 'epsilon-quant', 'epsilon-parallel',
                               'rcv1-th', 'rcv1-quant', 'rcv1-parallel']

    # dataset
    if args.experiment.startswith('epsilon'):
        dataset = os.path.expanduser('/mlodata1/jb/data/epsilon_normalized_1.pickle')
        n, d = 400000, 2000
    elif args.experiment.startswith('rcv1'):
        dataset = os.path.expanduser('/mlodata1/jb/data/rcv1-test-1.pickle')
        n, d = 677399, 47236

    # parameters to evaluate
    if args.experiment == 'epsilon-th':
        params = [
            Parameters(name="full-sgd", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2'),
            Parameters(name="full-sgd-no-shift", num_epoch=3, lr_type='decay', initial_lr=2, tau=1,
                       regularizer=1 / n, estimate='(t+tau)^2'),
            Parameters(name="top1", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True, take_top=True),
            Parameters(name="top1-no-shift", num_epoch=3, lr_type='decay', initial_lr=2, tau=1,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True, take_top=True),
            Parameters(name="rand1", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True),
            Parameters(name="rand1-no-shift", num_epoch=3, lr_type='decay', initial_lr=2, tau=1,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True),
            Parameters(name="rand2", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=2, with_memory=True),
            Parameters(name="rand2-no-shift", num_epoch=3, lr_type='decay', initial_lr=2, tau=1,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=2, with_memory=True),
            Parameters(name="rand3", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=3, with_memory=True),
            Parameters(name="rand3-no-shift", num_epoch=3, lr_type='decay', initial_lr=2, tau=1,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=3, with_memory=True),
        ]
    elif args.experiment == 'epsilon-quant':
        params = [
            Parameters(name="full-sgd", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2'),
            Parameters(name="qsgd-8bits", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2', qsgd_s=2 ** 8),
            Parameters(name="qsgd-4bits", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2', qsgd_s=2 ** 4),
            Parameters(name="qsgd-2bits", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2', qsgd_s=2 ** 2),
            Parameters(name="qsgd-sqrt-d-bits", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2', qsgd_s=44),
            Parameters(name="top1", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True, take_top=True),
            Parameters(name="rand1", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True),
        ]
    elif args.experiment == 'epsilon-parallel':
        models = [
            lambda n_cores: Parameters(name="rand1", num_epoch=5, lr_type='constant', initial_lr=.05, n_cores=n_cores,
                                       regularizer=1 / n, take_k=1, with_memory=True, estimate='final'),
            lambda n_cores: Parameters(name="top1", num_epoch=5, lr_type='constant', initial_lr=.05, n_cores=n_cores,
                                       regularizer=1 / n, take_k=1, take_top=True, with_memory=True, estimate='final'),
            lambda n_cores: Parameters(name="hogwild", num_epoch=5, lr_type='constant', initial_lr=.05, n_cores=n_cores,
                                       regularizer=1 / n, estimate='final'),
        ]
        cores = [1, 2, 3, 5, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        baseline = 0.305

    elif args.experiment == 'rcv1-th':
        params = [
            Parameters(name="full-sgd", num_epoch=3, lr_type='decay', initial_lr=2, tau=10,
                       regularizer=1 / n, estimate='(t+tau)^2'),
            Parameters(name="top10", num_epoch=3, lr_type='decay', initial_lr=2, tau=10 * d / 10,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=10, with_memory=True, take_top=True),
            Parameters(name="top10-no-shift", num_epoch=3, lr_type='decay', initial_lr=2, tau=10,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=10, with_memory=True, take_top=True),
            Parameters(name="rand10", num_epoch=3, lr_type='decay', initial_lr=2, tau=10 * d / 10,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=10, with_memory=True),
            Parameters(name="rand10-no-shift", num_epoch=3, lr_type='decay', initial_lr=2, tau=10,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=10, with_memory=True),
            Parameters(name="rand20", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=20, with_memory=True),
            Parameters(name="rand20-no-shift", num_epoch=3, lr_type='decay', initial_lr=2, tau=10,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=20, with_memory=True),
            Parameters(name="rand30", num_epoch=3, lr_type='decay', initial_lr=2, tau=d,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=30, with_memory=True),
            Parameters(name="rand30-no-shift", num_epoch=3, lr_type='decay', initial_lr=2, tau=10,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=30, with_memory=True),
        ]
    elif args.experiment == 'rcv1-quant':
        params = [
            Parameters(name="full-sgd", num_epoch=3, lr_type='decay', initial_lr=2, tau=10 * d,
                       regularizer=1 / n, estimate='(t+tau)^2'),
            Parameters(name="qsgd-8bits", num_epoch=3, lr_type='decay', initial_lr=2, tau=10 * d,
                       regularizer=1 / n, estimate='(t+tau)^2', qsgd_s=2 ** 8),
            Parameters(name="qsgd-4bits", num_epoch=3, lr_type='decay', initial_lr=2, tau=10 * d,
                       regularizer=1 / n, estimate='(t+tau)^2', qsgd_s=2 ** 4),
            Parameters(name="qsgd-2bits", num_epoch=3, lr_type='decay', initial_lr=2, tau=10 * d,
                       regularizer=1 / n, estimate='(t+tau)^2', qsgd_s=2 ** 2),
            Parameters(name="qsgd-sqrt-d-bits", num_epoch=3, lr_type='decay', initial_lr=2, tau=10 * d,
                       regularizer=1 / n, estimate='(t+tau)^2', qsgd_s=217),
            Parameters(name="top1", num_epoch=3, lr_type='decay', initial_lr=2, tau=10 * d,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True, take_top=True),
            Parameters(name="rand1", num_epoch=3, lr_type='decay', initial_lr=2, tau=10 * d,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=1, with_memory=True),
            Parameters(name="top10", num_epoch=3, lr_type='decay', initial_lr=2, tau=10 * d,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=10, with_memory=True, take_top=True),
            Parameters(name="rand10", num_epoch=3, lr_type='decay', initial_lr=2, tau=10 * d,
                       regularizer=1 / n, estimate='(t+tau)^2', take_k=10, with_memory=True),
        ]
    elif args.experiment == 'rcv1-parallel':
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
        baseline = 0.101

    if 'parallel' in args.experiment:
        run_parallel_experiment(args.directory, dataset, models, cores, baseline, repeat=3)
    else:
        run_experiment(args.directory, dataset, params, nproc=args.nproc)
