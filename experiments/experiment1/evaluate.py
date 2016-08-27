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
    Hyper = experiment_util.get_toy_hyper()
    extra_model_args = tuple()

    model_names = {  # mapped to optimization type
        "baselinedet": "ml",
        "baseline": "annealing",
        'mixture': "annealing",
        'planarflow': "annealing",
        'planarflowdet': "annealing",
        'radialflow': "annealing",
        'radialflowdet': "annealing",
    }

    dim = 1 if "1d" in datasetname else 2
    # def key(fn, path):
    #     return "%id" % dim in fn

    sampler = experiment_toy_models.toy_likelihood(dim=dim).function()

    def data_gen(hyper):
        hyper.dim = dim
        x_true = np.array([hyper.x_true] * hyper.dim, dtype=theano.config.floatX)
        _Z = np.array([sampler(x_true) for n in range(1000)], dtype=theano.config.floatX)
        Z, TZ = cross_validation.train_test_split(_Z, test_size=0.1)  # 10% test used in paper
        Z, VZ = cross_validation.train_test_split(Z, test_size=0.1)  # 20% validation used in paper
        data = None, Z, None, VZ, None, TZ  # None represents X data
        return data, experiment_util.RMSE

else:
    Hyper = experiment_util.get_hyper()

    data, error_func = experiment_util.load_and_preprocess_data(datasetname)
    X, Z, VX, VZ, TX, TZ = data
    example_input = X[0]
    example_output = Z[0]
    output_transfer = "softmax" if datasetname == "mnist" else "identity"
    extra_model_args = example_input, example_output, output_transfer

    model_names = {  # sorted by optimization type
        'baselinedet': "ml",
        'baselinedetplus': "ml",
        'baseline': "annealing",
        'baselineplus': "annealing",
        'mixture': "annealing",
        'planarflow': "annealing",
        'planarflowdet': "annealing",
        'radialflow': "annealing",
        'radialflowdet': "annealing",
    }

    def data_gen(hyper):
        return experiment_util.load_and_preprocess_data(hyper.datasetname)

# compute everything:
print "n_trials", n_trials
print "datasetname", datasetname

model_prefixes = model_names.keys()
best_hyper = eva.get_best_hyper(["withpercent"], Hyper, model_prefixes,
                                test_suffix=["best_val_loss"], key_files=lambda fn, p: datasetname in fn)

print "---------------------------------------------------------"
pprint(eva.fmap_results(lambda r: r[0], best_hyper))  # To see validation performance and whether it makes sense to sample these
print "---------------------------------------------------------"
# both methods are saving there results, also intermediately
pm.RNG = NestedNamespace(tm.PooledRandomStreams(pool_size=int(1e8)), RandomStreams())
test_results = eva.compute_test_results(best_hyper, data_gen,
                                        optimization_type=model_names, filepath=filepath_tests,
                                        extra_model_args=extra_model_args, n_trials=n_trials)
# Sample the approximate posterior distribution for evaluation
best_hyper_samples = eva.sample_best_hyper(best_hyper, test_results, filepath=filepath_samples)
