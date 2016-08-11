# coding: utf-8
from __future__ import division

import operator as op
import os, platform, sys
from ast import literal_eval
from pprint import pformat, pprint

import itertools
import numpy as np
import csv

from schlichtanders.mycontextmanagers import ignored
from schlichtanders.myobjects import NestedNamespace, Namespace

import theano_models as tm
import theano_models.deterministic_models as dm
import theano_models.probabilistic_models as pm
from theano.tensor.shared_randomstreams import RandomStreams
from copy import copy
import warnings

from experiment_util import log_exceptions, setup_sqlite, hyper_init_dict, hyper_init_random, \
    optimize, load_and_preprocess_data, hyper_init_several, hyper_init_mnist
import experiment_models

inf = float("inf")
warnings.filterwarnings("ignore", category=DeprecationWarning)

__file__ = os.path.realpath(__file__)
if platform.system() == "Windows":
    from schlichtanders.myos import replace_unc
    __file__ = replace_unc(__file__)
__path__ = os.path.dirname(__file__)
__parent__ = os.path.dirname(__path__)

# defaults:
foldername = "experiment"
filename = "several"
datasetname = "boston"
#overwrite as far as given:
if len(sys.argv) > 3:
    foldername, filename, datasetname = sys.argv[1:4]
elif len(sys.argv) > 2:
    filename, datasetname = sys.argv[1:3]
elif len(sys.argv) > 1:
    datasetname = sys.argv[1]

with ignored(OSError):
    os.mkdir(os.path.join(__path__, foldername))
filepath = os.path.join(__path__, foldername, '%s.db' % filename)
errorfilepath = os.path.join(__path__, foldername, '%s_errors.txt' % filename)
csvpath = os.path.join(__path__, 'good_parameters.csv')


tm.inputting_references.update(['to_be_randomized'])
pm.RNG = NestedNamespace(tm.PooledRandomStreams(pool_size=int(5e8)), RandomStreams())

# # Data
#     # datasetnames = ["boston", "concrete", "energy", "kin8nm", "naval", "powerplant", "protein", "winered", "yacht", "year"]
#     datasetnames = ["boston", "concrete", "energy", "kin8nm", "powerplant", "winered", "yacht"]

# "naval" is not possible as standardization leads to a nan column. Hence we do not know which to use
# datasetname = "concrete"


model_names = { # sorted by optimization type
    "ml": ['baselinedet', 'baselinedetplus'],
    "annealing": ['baseline', 'baselineplus', 'mixture',
                  'planarflow', 'planarflowdet', 'radialflow', 'radialflowdet'],
    "ml_exp_average": ['mixtureml', 'planarflowml', 'radialflowml'],
}

# Hyperparameters
# ===============

Hyper, sql_session = setup_sqlite(model_prefixes=reduce(op.add, model_names.values()), abs_path_sqlite=filepath)
hyper_init = hyper_init_mnist if datasetname=="mnist" else hyper_init_several


# Data
# ====
data, error_func = load_and_preprocess_data(datasetname)
X, Z, VX, VZ, TX, TZ = data
example_input = X[0]
example_outpt = Z[0]
output_transfer = "softmax" if datasetname=="mnist" else "identity"

# TODO the above randomly splits the dataset, which should be averaged out ideally... however with same initial parameters... that makes it difficult

# Main Loop
# =========
def optimize_all(hyper):
    error_info = {k: v for k, v in hyper.__dict__.iteritems() if k[:4] not in ["base", "plana", "mixt", "radi"]}
    for optimization_type in model_names:
        for model_name in model_names[optimization_type]:
            # reset hard the saved models, to prevent any possible interaction (hard to debug)
            dm.InvertibleModel.INVERTIBLE_MODELS = []
            tm.Model.all_models = []

            with log_exceptions(errorfilepath, model_name, error_info):
                model, loss, flat = getattr(experiment_models, model_name)(
                    hyper, example_input, example_outpt, output_transfer)
                optimize(  # updates hyper inplace with results
                    prefix=model_name, data=data, hyper=hyper, model=model, loss=loss,
                    parameters=flat, error_func=error_func, type=optimization_type
                )
                sql_session.commit()

sample_new = False
# randomly sample new hyperparameters
if sample_new:
    while True:
        for i in range(3):  # repeat, taking slightly different starting parameters each time
            if i == 0:
                hyper = Hyper(datasetname)
                hyper_init_random(hyper)
                hyper_init(hyper)
                hyper_dict = copy(hyper.__dict__)
                pprint(hyper_dict)
                sql_session.add(hyper)
            else:
                hyper = Hyper(datasetname)  # new hyper with same parameters
                hyper_init_dict(hyper, hyper_dict)
                sql_session.add(hyper)

            optimize_all(hyper)


# use already found parameters which seem good
else:
    good_parameters = []
    with open(csvpath, "r") as f:
        reader = csv.DictReader(f, quoting=csv.QUOTE_NONE)
        for row in reader:
            # this should not be updated in hyper
            # (double quote as quoting=csv.QUOTE_NONNUMERIC was used to create this csv)
            del row['"datasetname"']
            # TODO if the first runs through, delete also n_normflows, as we want to have all of them
            good_parameters.append(
                {literal_eval(k): literal_eval(v) for k, v in row.iteritems()}
            )  # evaluate everything, this version also distinguishes ints/floats

    for i in itertools.count():  # repeat the set
        print "ROUND %i" % i
        print "=========================================="

        for ip, params in enumerate(good_parameters):
            print "parameterset %i" % ip
            print "----------------------------------------------"
            hyper = Hyper(datasetname)
            hyper_init_random(hyper)
            hyper_init(hyper)
            hyper_init_dict(hyper, params)
            if datasetname != "mnist":  # we need the n_normflow parameter for this to be valid
                hyper.units_per_layer_plus = hyper.units_per_layer + hyper.units_per_layer * (hyper.n_normflows * 2)
            sql_session.add(hyper)

            optimize_all(hyper)