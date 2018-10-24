import multiprocessing as mp
import queue
import time
from multiprocessing import sharedctypes

import numpy as np
from numpy import ctypeslib
from scipy.sparse import isspmatrix
from scipy.special import expit as sigmoid

from base_logistic import BaseLogistic
from constants import INIT_WEIGHT_STD
from memory import GradientMemory
from parameters import Parameters

TIMEOUT = 1
LOSS_PER_EPOCH = 10


class LogisticParallelSGD(BaseLogistic):
    """
    2 classes logistic regression on dense dataset.
    X: (num_samples, num_features)
    y: (num_features, ) 0, 1 labels
    """

    def __init__(self, params: Parameters):
        super().__init__(params)
        self.params = params
        self.w = None

        self.epoch_callback = None
        self.print = False

    def fit_until(self, X, y, num_samples, num_features, baseline=None):
        # num_samples, num_features = X.shape
        p = self.params

        if self.w is None:
            self.w = np.random.normal(0, INIT_WEIGHT_STD, size=(num_features,))

        def worker_fit(id_w, num_workers, X_w, y_w, weights_w, shape, indices, results, params_w, stopper):
            # reconstruct numpy shared array
            num_samples, num_features = shape
            weights_w = ctypeslib.as_array(weights_w)
            weights_w.shape = (num_features,)

            if not isspmatrix(X_w):
                X_w = ctypeslib.as_array(X_w)
                X_w.shape = (num_samples, num_features)
                y_w = ctypeslib.as_array(y_w)
                y_w.shape = (num_samples,)

            memory = GradientMemory(take_k=params_w.take_k, take_top=params_w.take_top,
                                    with_memory=params_w.with_memory)

            if id_w == 0:
                losses = np.zeros(params_w.num_epoch * LOSS_PER_EPOCH + 1)
                losses[0] = self.loss(X, y)
                start_time = time.time()
                last_printed = 0

            loss_every = num_samples // LOSS_PER_EPOCH

            for epoch in range(params_w.num_epoch):
                for iteration in range(id_w, num_samples, num_workers):
                    # worker 0 gave stop signal, reached accuracy
                    if stopper.value:
                        return
                    sample_idx = indices[epoch][iteration]

                    lr = self.lr(epoch, iteration, num_samples, num_features)

                    x = X_w[sample_idx]

                    if isspmatrix(x):
                        x = np.array(x.todense()).squeeze(0)
                    minus_grad = y[sample_idx] * x * sigmoid(-y[sample_idx] * np.dot(x, self.w))
                    # minus_grad = - x * (pred_proba - y_w[sample_idx])
                    if params_w.regularizer:
                        minus_grad -= 2 * params_w.regularizer * weights_w

                    sparse = params_w.take_k and (params_w.take_k < num_features)
                    # next_real -= 1
                    lr_minus_grad = memory(lr * minus_grad, sparse=sparse)  # , no_apply=(next_real != 0))
                    # if next_real == 0:
                    #     next_real = params_w.real_update_every

                    if sparse:
                        weights_w[lr_minus_grad[0]] += lr_minus_grad[1]
                    else:
                        weights_w += lr_minus_grad

                    if id_w == 0 and num_samples * epoch + iteration - last_printed >= loss_every:
                        last_printed = num_samples * epoch + iteration
                        timing = time.time() - start_time
                        loss = self.loss(X, y)
                        losses[epoch * LOSS_PER_EPOCH + (iteration // loss_every) + 1] = loss
                        print("epoch {} iter {} loss {} time {}s".format(
                            epoch, iteration, loss, timing))

                        if baseline and loss <= baseline:
                            stopper.value = True
                            results['epoch'] = epoch
                            results['losses'] = losses
                            results['iteration'] = iteration
                            results['timing'] = timing
                            return

            # if failed to converge...
            if id_w == 0:
                results['epoch'] = epoch
                results['losses'] = losses
                results['iteration'] = iteration
                results['timing'] = time.time() - start_time

        with mp.Manager() as manager:
            results = manager.dict()
            stopper = manager.Value('b', False)

            indices = np.zeros((p.num_epoch, num_samples), dtype=int)
            for i in range(p.num_epoch):
                indices[i] = np.arange(num_samples)
                np.random.shuffle(indices[i])

            weights_w = sharedctypes.RawArray('d', self.w)
            self.w = ctypeslib.as_array(weights_w)
            self.w.shape = (num_features,)

            if isspmatrix(X):
                X_w = X
                y_w = y
            else:
                X_w = sharedctypes.RawArray('d', np.ravel(X))
                y_w = sharedctypes.RawArray('d', y)

            processes = [mp.Process(target=worker_fit,
                                    args=(
                                        i, p.n_cores, X_w, y_w, weights_w, X.shape, indices, results, self.params,
                                        stopper))
                         for i in range(p.n_cores)]

            for p in processes:
                p.start()
            for i, p in enumerate(processes):
                p.join()

            print(results)
            return results['timing'], results['epoch'], results['iteration'], results['losses']

    def fit(self, X, y, num_samples, num_features, loss_per_epoch=10):
        p = self.params

        if self.w is None:
            self.w = np.random.normal(0, INIT_WEIGHT_STD, size=(num_features,))

        def worker_fit(id_w, num_workers, X_w, y_w, weights_w, shape, indices, counter, start_barrier, params_w):
            assert params_w.regularizer is not None
            # reconstruct numpy shared array
            num_samples, num_features = shape
            weights_w = ctypeslib.as_array(weights_w)
            weights_w.shape = (num_features,)

            if not isspmatrix(X_w):
                X_w = ctypeslib.as_array(X_w)
                X_w.shape = (num_samples, num_features)
                y_w = ctypeslib.as_array(y_w)
                y_w.shape = (num_samples,)

            memory = GradientMemory(take_k=params_w.take_k, take_top=params_w.take_top,
                                    with_memory=params_w.with_memory)

            start_barrier.wait()
            while True:
                with counter.get_lock():
                    idx = counter.value
                    counter.value += 1

                if idx >= num_samples * params_w.num_epoch:
                    break

                sample_idx = indices[idx]
                epoch = idx // num_samples
                iteration = idx % num_samples
                lr = self.lr(epoch, iteration, num_samples, num_features)

                x = X_w[sample_idx]

                if isspmatrix(x):
                    minus_grad = -1. * params_w.regularizer * weights_w
                    sparse_minus_grad = y[sample_idx] * x * sigmoid(-y[sample_idx] * x.dot(weights_w).squeeze(0))
                    minus_grad[sparse_minus_grad.indices] += sparse_minus_grad.data

                else:
                    minus_grad = y[sample_idx] * x * sigmoid(-y[sample_idx] * x.dot(weights_w))
                    minus_grad -= params_w.regularizer * weights_w

                sparse = params_w.take_k and (params_w.take_k < num_features)
                lr_minus_grad = memory(lr * minus_grad, sparse=sparse)

                if sparse:
                    weights_w[lr_minus_grad[0]] += lr_minus_grad[1]
                else:
                    weights_w += lr_minus_grad

        with mp.Manager() as manager:
            counter = mp.Value('i', 0)
            start_barrier = manager.Barrier(p.n_cores + 1)  # wait all worker and the monitor to be ready

            indices = np.zeros((p.num_epoch, num_samples), dtype=int)
            for i in range(p.num_epoch):
                indices[i] = np.arange(num_samples)
                np.random.shuffle(indices[i])
            indices = indices.flatten()

            weights_w = sharedctypes.RawArray('d', self.w)
            self.w = ctypeslib.as_array(weights_w)
            self.w.shape = (num_features,)

            if isspmatrix(X):
                X_w = X
                y_w = y
            else:
                X_w = sharedctypes.RawArray('d', np.ravel(X))
                y_w = sharedctypes.RawArray('d', y)

            processes = [mp.Process(target=worker_fit,
                                    args=(
                                        i, p.n_cores, X_w, y_w, weights_w, X.shape, indices, counter, start_barrier,
                                        self.params))
                         for i in range(p.n_cores)]

            for p in processes:
                p.start()

            # monitor the progress
            print_every = num_samples // loss_per_epoch
            next_print = 0

            # loss computing on another thread through the queue
            stop = manager.Value('b', False)
            w_queue = mp.Queue()
            results = manager.dict()

            def loss_computer(q, regularizer, res, stop):  # should be stoppable
                print('start loss computer')
                losses = []
                iters = []
                timers = []

                while not q.empty() or not stop.value:
                    try:
                        epoch, iter_, total_iter, chrono, w = q.get(block=True, timeout=1)
                    except queue.Empty:
                        # print('empty queue')
                        continue
                    # print('dequeue', epoch, iter_)

                    loss = np.sum(np.log(1 + np.exp(-y * (X @ w)))) / X.shape[0]
                    if regularizer is not None:
                        loss += regularizer * np.square(w).sum() / 2

                    timers.append(chrono)
                    losses.append(loss)
                    iters.append(total_iter)

                    print("epoch {} iteration {} loss {} time {}s".format(epoch, iter_, loss, chrono))

                res['losses'] = np.array(losses)
                res['iters'] = np.array(iters)
                res['timers'] = np.array(timers)

            start_barrier.wait()
            start_time = time.time()

            loss_computer = mp.Process(target=loss_computer, args=(w_queue, self.params.regularizer, results, stop))
            loss_computer.start()

            while counter.value < self.params.num_epoch * num_samples:
                if counter.value > next_print:
                    w_copy = (self.w_estimate if self.w_estimate is not None else self.w).copy()
                    epoch = next_print // num_samples
                    iter_ = next_print % num_samples
                    chrono = time.time() - start_time

                    w_queue.put((epoch, iter_, next_print, chrono, w_copy))
                    # print('enqueue', epoch, iter_)

                    next_print += print_every
                else:
                    time.sleep(.1)

            stop.value = True  # stop the loss computer
            for i, p in enumerate(processes):
                p.join()

            loss_computer.join()

            print(results)
            return results['iters'], results['timers'], results['losses']
