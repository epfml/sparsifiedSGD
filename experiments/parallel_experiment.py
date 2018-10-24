import argparse
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from logistic_parallel import LogisticParallelSGD
from parameters import Parameters
from utils import pickle_it

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_file', type=str, help="pickle file of the dataset (X, y)")
parser.add_argument('--output_directory', type=str)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_cores', type=int)
parser.add_argument('--model', choices=['top', 'rand', 'sgd'])
parser.add_argument('--k', type=int)
parser.add_argument('--initial_lr', type=float)
parser.add_argument('--lr', choices=['bottou'])

args = parser.parse_args()
if args.k and args.model == 'sgd':
    args.k = None
    print('k ignored')

# create uniquely named suffix for the results
name = "{}{}-cores{}".format(args.model, args.k or "", args.num_cores)
directory = os.path.join(args.output_directory, name)
i = 0
while os.path.exists("{}-{}".format(directory, i)):
    i += 1

directory = "{}-{}".format(directory, i)
os.makedirs(directory)
print('will save results in "{}"'.format(directory))

with_memory = args.model in ['top', 'rand']
print('use memory', with_memory)
use_top = (args.model == 'top')
print('use top coordinates', use_top)

print('load dataset')
with open(args.dataset_file, 'rb') as f:
    X, y = pickle.load(f)

n, d = X.shape

model = Parameters(num_epoch=args.num_epochs,
                   lr_type=args.lr,
                   initial_lr=args.initial_lr,
                   n_cores=args.num_cores,
                   regularizer=1. / n,
                   take_k=args.k,
                   take_top=args.model is 'top',
                   with_memory=with_memory,
                   estimate='final')

print('start experiment')

m = LogisticParallelSGD(model)
iters, timers, losses = m.fit(X, y,
                              num_features=X.shape[1],
                              num_samples=X.shape[0])

pickle_it(iters, 'iters', directory)
pickle_it(timers, 'timers', directory)
pickle_it(losses, 'losses', directory)

print('results saved in "{}"'.format(directory))
