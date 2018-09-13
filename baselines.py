import argparse
import os
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier

from utils import pickle_it

"""Arguments"""

parser = argparse.ArgumentParser()
parser.add_argument('data', type=str)
parser.add_argument('output', type=str)

EPSILON_NAME = "epsilon.pickle"
RCV1_NAME = "rcv1.pickle"

args = parser.parse_args()
if not os.path.exists(args.output):
    print('create {}'.format(args.output))
    os.makedirs(args.output)

baselines = {}


def loss(clf, X, y, reg):
    baseline_loss = np.sum(np.log(1 + np.exp(-y * (X @ clf.coef_.transpose()).squeeze()))) / X.shape[0]
    baseline_loss += reg / 2 * np.sum(np.square(clf.coef_))
    return baseline_loss


""" RCV1 test"""
print('RCV1-test')
with open(os.path.join(args.data, RCV1_NAME), 'rb') as f:
    X, y = pickle.load(f)

reg = 1 / X.shape[0]
clf = SGDClassifier(tol=1e-4, loss='log', penalty='l2', alpha=reg, fit_intercept=False)
clf.fit(X, y)
l = loss(clf, X, y, reg)
print("loss: {}".format(l))
print("train accuracy: {}".format(clf.score(X, y)))
baselines['RCV1-test'] = l

""" EPSILON """

print('epsilon')
with open(os.path.join(args.data, EPSILON_NAME), 'rb') as f:
    X, y = pickle.load(f)

reg = 1 / X.shape[0]
clf = SGDClassifier(tol=1e-4, loss='log', penalty='l2', alpha=reg)
clf.fit(X, y)
l = loss(clf, X, y, reg)
print("loss: {}".format(l))
print("train accuracy: {}".format(clf.score(X, y)))
baselines['epsilon'] = l

""" Pickle """
print('baselines', baselines)
pickle_it(baselines, 'baselines', args.output)
