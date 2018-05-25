import numpy as np

"""
Quantization scheme for QSGD
Follows Alistarh, 2017 (https://arxiv.org/abs/1610.02132) but without the compression scheme.
"""


def quantize(x, d):
    """quantize the tensor x in d level on the absolute value coef wise"""
    norm = np.sqrt(np.sum(np.square(x)))
    level_float = d * np.abs(x) / norm
    previous_level = np.floor(level_float)
    is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    return np.sign(x) * norm * new_level / d
