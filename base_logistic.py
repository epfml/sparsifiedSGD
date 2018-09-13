import numpy as np
from scipy.special import expit as sigmoid

from parameters import Parameters


class BaseLogistic:
    def __init__(self, params: Parameters):
        self.params = params
        self.w_estimate = None
        self.w = None

    def lr(self, epoch, iteration, num_samples, d):
        p = self.params
        t = epoch * num_samples + iteration
        if p.lr_type == 'constant':
            return p.initial_lr
        if p.lr_type == 'epoch-decay':
            return p.initial_lr * (p.epoch_decay_lr ** epoch)
        if p.lr_type == 'decay':
            return p.initial_lr / (p.regularizer * (t + p.tau))
        if p.lr_type == 'bottou':
            return p.initial_lr / (1 + p.initial_lr * p.regularizer * t)

    def loss(self, X, y):
        w = self.w_estimate if self.w_estimate is not None else self.w
        w = w.copy()
        p = self.params
        loss = np.sum(np.log(1 + np.exp(-y * (X @ w)))) / X.shape[0]
        if p.regularizer:
            loss += p.regularizer * np.square(w).sum() / 2
        return loss

    def predict(self, X):
        w = self.w_estimate if self.w_estimate is not None else self.w
        logits = X @ w
        pred = 1 * (logits >= 0.)
        return pred

    def predict_proba(self, X):
        w = self.w_estimate if self.w_estimate is not None else self.w
        logits = X @ w
        return sigmoid(logits)

    def score(self, X, y):
        w = self.w_estimate if self.w_estimate is not None else self.w
        logits = X @ w
        pred = 2 * (logits >= 0.) - 1
        acc = np.mean(pred == y)
        return acc

    def update_estimate(self, t):
        t = int(t)  # to avoid overflow with np.int32
        p = self.params
        if p.estimate == 'final':
            self.w_estimate = self.w
        elif p.estimate == 'mean':
            rho = 1 / (t + 1)
            self.w_estimate = self.w_estimate * (1 - rho) + self.w * rho
        elif p.estimate == 't+tau':
            rho = 2 * (t + p.tau) / ((1 + t) * (t + 2 * p.tau))
            self.w_estimate = self.w_estimate * (1 - rho) + self.w * rho
        elif p.estimate == '(t+tau)^2':
            rho = 6 * ((t + p.tau) ** 2) / ((1 + t) * (6 * (p.tau ** 2) + t + 6 * p.tau * t + 2 * (t ** 2)))
            self.w_estimate = self.w_estimate * (1 - rho) + self.w * rho

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.params)

    def __repr__(self):
        return str(self)
