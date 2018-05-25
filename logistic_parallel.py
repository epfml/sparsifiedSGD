import multiprocessing as mp
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

            next_real = params_w.real_update_every
            for epoch in range(params_w.num_epoch):
                for iteration in range(id_w, num_samples, num_workers):
                    # worker 0 gave stop signal, reached accuracy
                    if stopper.value:
                        return
                    sample_idx = indices[epoch][iteration]

                    lr = self.lr(epoch, iteration, num_samples, num_features)

                    pred_logits = X_w[sample_idx] @ weights_w
                    pred_proba = sigmoid(pred_logits)
                    x = X_w[sample_idx]

                    if isspmatrix(x):
                        x = np.array(x.todense()).squeeze(0)
                    minus_grad = y[sample_idx] * x * sigmoid(-y[sample_idx] * np.dot(x, self.w))
                    # minus_grad = - x * (pred_proba - y_w[sample_idx])
                    if params_w.regularizer:
                        minus_grad -= 2 * params_w.regularizer * weights_w

                    sparse = params_w.take_k and (params_w.take_k < num_features)
                    next_real -= 1
                    lr_minus_grad = memory(lr * minus_grad, sparse=sparse, no_apply=(next_real != 0))
                    if next_real == 0:
                        next_real = params_w.real_update_every

                    if lr_minus_grad is not None:
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
                                    i, p.n_cores, X_w, y_w, weights_w, X.shape, indices, results, self.params, stopper))
                         for i in range(p.n_cores)]

            for p in processes:
                p.start()
            for i, p in enumerate(processes):
                p.join()

            print(results)
            return results['timing'], results['epoch'], results['iteration'], results['losses']
