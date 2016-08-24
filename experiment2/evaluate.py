from __future__ import division
from ast import literal_eval

from sklearn import cross_validation
from pprint import pprint

import os, platform, sys
import warnings

import experiment_toy_models
import experiment_util
from schlichtanders.mycontextmanagers import ignored
from schlichtanders.myobjects import NestedNamespace
import evaluation_util as eva

import numpy as np
import theano
import theano_models as tm
from theano.tensor.shared_randomstreams import RandomStreams
import theano_models.probabilistic_models as pm


__file__ = os.path.realpath(__file__)
if platform.system() == "Windows":
    from schlichtanders.myos import replace_unc
    __file__ = replace_unc(__file__)
__path__ = os.path.dirname(__file__)
__parent__ = os.path.dirname(__path__)

sys.path.append(__parent__)
import experiment1.experiment_util as ex1util

folders_parameters = [["experiment1","withpercent"], ["experiment1","first_useful_hyperparameter_search"], ['experiment1','run_windows']]
folders_parameters = [os.path.join(__parent__, *fp) for fp in folders_parameters]

warnings.filterwarnings("ignore", category=DeprecationWarning)
inf = float('inf')


foldername = "evaluation"
datasetname = "toy1d"
n_trials = 10

#overwrite as far as given:
if len(sys.argv) > 3:
    foldername, datasetname, n_trials = sys.argv[1:4]
    n_trials = int(literal_eval(n_trials))
if len(sys.argv) > 2:
    datasetname, n_trials = sys.argv[1:3]
    n_trials = int(literal_eval(n_trials))
elif len(sys.argv) > 1:
    datasetname = sys.argv[1]

with ignored(OSError):
    os.mkdir(os.path.join(__path__, foldername))

filepath_tests = os.path.join(__path__, foldername, "%s_tests.pkl" % datasetname)
filepath_samples = os.path.join(__path__, foldername, "%s_samples.pkl" % datasetname)
# -------------------------------------------


# # Collect models and find best ones
# best_hyper = eva.get_best_hyper(["toy_windows", "toy_linux"], Hyper, model_prefixes, test_suffix=["best_val_loss"])
if "toy" in datasetname:
    dim = 1 if "1d" in datasetname else 2
    sampler = experiment_toy_models.toy_likelihood(dim=dim).function()
    def data_gen(hyper):
        x_true = np.array([0.0] * dim, dtype=theano.config.floatX)
        _Z = np.array([sampler(x_true) for _ in range(1000)], dtype=theano.config.floatX)
        Z, TZ = cross_validation.train_test_split(_Z, test_size=0.1)  # 10% test used in paper
        Z, VZ = cross_validation.train_test_split(Z, test_size=0.1)  # 20% validation used in paper
        data = None, Z, None, VZ, None, TZ  # None represents X data
        return data, experiment_util.RMSE
else:
    def data_gen(hyper):
        return experiment_util.load_and_preprocess_data(hyper.datasetname)


# compute everything:
print "n_trials", n_trials
print "datasetname", datasetname

best_hypers = eva.get_best_hyper_autofix(datasetname, folders_parameters)

print "---------------------------------------------------------"
pprint([(h.modelname, h.n_normflows, h.percent, h.best_val_loss) for h in best_hypers])  # To see validation performance and whether it makes sense to sample these
print "---------------------------------------------------------"
# both methods are saving there results, also intermediately
pm.RNG = NestedNamespace(tm.PooledRandomStreams(pool_size=int(1e8)), RandomStreams())
best_hyper_tests = eva.compute_test_results(best_hypers, data_gen, filepath=filepath_tests, n_trials=n_trials)
# Sample the approximate posterior distribution for evaluation
best_hyper_samples = eva.sample_best_hyper(best_hyper_tests, filepath=filepath_samples)
