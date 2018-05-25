import time

import numpy as np
from scipy.sparse import isspmatrix
from scipy.special import expit as sigmoid

from base_logistic import BaseLogistic
from constants import INIT_WEIGHT_STD, LOSS_PER_EPOCH
from memory import GradientMemory
from parameters import Parameters


class LogisticSGD(BaseLogistic):
    """
    2 classes logistic regression on dense dataset.
    X: (num_samples, num_features)
    y: (num_features, ) 0, 1 labels
    """

    def __init__(self, params: Parameters):
        super().__init__(params)
        self.w = None
        self.w_estimate = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        p = self.params

        losses = np.zeros(p.num_epoch + 1)

        if self.w is None:
            self.w = np.random.normal(0, INIT_WEIGHT_STD, size=(num_features,))
            self.w_estimate = np.copy(self.w)

        memory = GradientMemory(take_k=p.take_k, take_top=p.take_top, with_memory=p.with_memory, qsgd_s=p.qsgd_s)

        shuffled_indices = np.arange(num_samples)

        # epoch 0 loss evaluation
        losses[0] = self.loss(X, y)

        train_start = time.time()

        compute_loss_every = int(X.shape[0] / LOSS_PER_EPOCH)
        all_losses = np.zeros(LOSS_PER_EPOCH * p.num_epoch + 1)

        for epoch in np.arange(p.num_epoch):
            np.random.shuffle(shuffled_indices)

            for iteration in range(num_samples):
                t = epoch * num_samples + iteration
                if t % compute_loss_every == 0:
                    loss = self.loss(X, y)
                    print('{} t {} epoch {} iter {} loss {}'.format(p, t, epoch, iteration, loss))
                    all_losses[t // compute_loss_every] = loss

                sample_idx = shuffled_indices[iteration]

                lr = self.lr(epoch, iteration, num_samples, num_features)

                # pred_logits = X[sample_idx] @ self.w
                # pred_proba = sigmoid(pred_logits)
                x = X[sample_idx]

                if isspmatrix(x):
                    x = np.array(x.todense()).squeeze(0)
                minus_grad = y[sample_idx] * x * sigmoid(-y[sample_idx] * np.dot(x, self.w))
                # minus_grad = - x * (pred_proba - y[sample_idx])
                if p.regularizer:
                    minus_grad -= 2 * p.regularizer * self.w

                lr_minus_grad = memory(lr * minus_grad)
                self.w += lr_minus_grad

                self.update_estimate(t)

            losses[epoch + 1] = self.loss(X, y)
            print("epoch {}: loss {} score {}".format(epoch, losses[epoch + 1], self.score(X, y)))

        print("Training took: {}s".format(time.time() - train_start))

        return losses, all_losses
