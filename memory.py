import numpy as np

import qsgd

"""
Gradient memory module

Keep unused gradient in memory and use them later.
Can be used with random or top k sparsifier.
"""


class GradientMemory:
    def __init__(self, take_k=None, take_top=False, with_memory=False, qsgd_s=None):
        self.with_memory = with_memory
        self.take_top = take_top
        self.take_k = take_k
        self.qsgd_s = qsgd_s
        self.m = None

    def __call__(self, g, sparse=False): # , no_apply=False):
        if self.qsgd_s:
            return qsgd.quantize(g, self.qsgd_s)

        if not self.take_k:
            return g

        if self.with_memory:
            # create the memory if does not exist
            if self.m is None:
                self.m = np.zeros(g.shape, dtype=np.float64)
            self.m += g
        else:
            self.m = g

        # for k < 1 sometimes no gradient is used from the memory
        # if no_apply:
        #     return None

        d = np.prod(self.m.shape)
        k = min(self.take_k, d)
        if self.take_top:
            indices = np.argpartition(np.abs(self.m.ravel()), -k)[-k:]
        else:
            indices = np.random.choice(d, k, replace=False)

        if not sparse:
            out_grad = np.zeros_like(self.m)
            out_grad[indices] = self.m[indices]
        else:
            out_grad = (indices, self.m[indices])
        self.m[indices] = 0.
        return out_grad
