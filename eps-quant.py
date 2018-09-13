# For normal SGD bottou final
# best is lr = 1.0
from parameters import Parameters
from logistic import LogisticSGD
from experiment import run_experiment

n = 400000
params = []

num_epoch=5
lr=1.
params.append(Parameters(name="full-sgd", num_epoch=num_epoch, lr_type='bottou', initial_lr=lr,
                   regularizer=1 / n, estimate='final'))
params.append(Parameters(name="qsgd-8bit", num_epoch=num_epoch, lr_type='bottou', initial_lr=lr,
                   regularizer=1 / n, estimate='final',
                        qsgd_s=2 ** 8))
params.append(Parameters(name="qsgd-4bit", num_epoch=num_epoch, lr_type='bottou', initial_lr=lr,
                   regularizer=1 / n, estimate='final',
                        qsgd_s=2 ** 4))
params.append(Parameters(name="qsgd-2bit", num_epoch=num_epoch, lr_type='bottou', initial_lr=lr,
                   regularizer=1 / n, estimate='final',
                        qsgd_s=2 ** 2))
params.append(Parameters(name="top1", num_epoch=num_epoch, lr_type='bottou', initial_lr=lr,
                   regularizer=1 / n, estimate='final',
                        take_k=1, take_top=True, with_memory=True))
params.append(Parameters(name="rand1", num_epoch=num_epoch, lr_type='bottou', initial_lr=lr,
                   regularizer=1 / n, estimate='final',
                        take_k=1, take_top=False, with_memory=True))
                  
run_experiment('eps-quantized', '/mlodata1/jb/data/epsilon_normalized_1.pickle', params, nproc=12)