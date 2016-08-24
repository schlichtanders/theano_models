# coding: utf-8
from __future__ import division

import operator as op
import os, platform, sys
import random
from ast import literal_eval
from pprint import pformat, pprint

import itertools
from sklearn import cross_validation

import numpy as np
import csv

from sqlalchemy import Column, Integer, Unicode, UnicodeText, String, PickleType, Float, Boolean
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


from schlichtanders.mycontextmanagers import ignored
from schlichtanders.myobjects import NestedNamespace, Namespace

import theano
import theano.tensor as T
import theano_models as tm
import theano_models.deterministic_models as dm
import theano_models.probabilistic_models as pm
from theano.tensor.shared_randomstreams import RandomStreams
from copy import copy
import warnings

from experiment_util import log_exceptions, hyper_init_dict, get_init_random, \
    optimize, load_and_preprocess_data, get_init_several, get_init_mnist, RMSE, get_toy_hyper
import experiment_toy_models

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
filename = "toy_example"
dim = 1
#overwrite as far as given:
if len(sys.argv) > 3:
    foldername, filename, dim = sys.argv[1:4]
    dim = int(literal_eval(dim))
elif len(sys.argv) > 2:
    foldername, filename = sys.argv[1:3]
elif len(sys.argv) > 1:
    foldername = sys.argv[1]


with ignored(OSError):
    os.mkdir(os.path.join(__path__, foldername))
filepath = os.path.join(__path__, foldername, '%s.db' % filename)
errorfilepath = os.path.join(__path__, foldername, '%s_errors.txt' % filename)
csvpath = os.path.join(__path__, 'good_parameters.csv')


model_names = { # sorted by optimization type
    "ml": ['baselinedet'],
    "annealing": ['baseline', 'mixture', 'planarflow', 'planarflowdet', 'radialflow', 'radialflowdet'],
}  # TODO mixture needed? not useful for testruns, however for modes really good (?)
model_prefixes = reduce(op.add, model_names.values())
model_prefixes = [p+"_" for p in model_prefixes]

# Hyperparameters
# ===============

Hyper = get_toy_hyper()
engine = create_engine('sqlite:///' + filepath)  # os.path.join(__path__, foldername, '%s.db' % filename)
Hyper.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
sql_session = Session()


# DATA
# ====

# pm.RNG = NestedNamespace(tm.PooledRandomStreams(pool_size=int(1e8)), RandomStreams())

# The function:
x_true = 0.0  # 0.15
_x_true = np.array([x_true]*dim, dtype=theano.config.floatX)

sampler = experiment_toy_models.toy_likelihood(dim=dim).function()
Z = np.array([sampler(_x_true) for n in range(1000)], dtype=theano.config.floatX)
Z, TZ = cross_validation.train_test_split(Z, test_size=0.1)  # 10% test used in paper
Z, VZ = cross_validation.train_test_split(Z, test_size=0.1)  # 20% validation used in paper
data = None, Z, None, VZ, None, TZ  # None represents X data
original_Z = Z
error_func = RMSE

# TODO the above randomly splits the dataset, which should be averaged out ideally... however with same initial parameters... that makes it difficult

# Main Loop
# =========
def optimize_all(hyper):
    data = None, Z, None, VZ, None, TZ  # None represents X data
    error_info = {k: v for k, v in hyper.__dict__.iteritems() if k[:4] not in ["base", "plana", "mixt", "radi"]}
    for optimization_type in model_names:
        for model_name in model_names[optimization_type]:
            # reset hard the saved models, to prevent any possible interaction (hard to debug)
            dm.InvertibleModel.INVERTIBLE_MODELS = []
            tm.Model.all_models = []

            with log_exceptions(errorfilepath, model_name, error_info):
                model, loss, flat, approx_posterior = getattr(experiment_toy_models, model_name)(hyper)
                optimize(  # updates hyper inplace with results
                    prefix=model_name, data=data, hyper=hyper, model=model, loss=loss,
                    parameters=flat, error_func=error_func, optimization_type=optimization_type
                )
                sql_session.commit()

sample_new = False
# randomly sample new hyperparameters
if sample_new:
    while True:
        # TODO loop over all n_normflows
        for ir in range(3):  # repeat, taking slightly different starting parameters each time
            if ir == 0:
                hyper = Hyper(x_true, dim)
                get_init_random(hyper)
                hyper_dict = copy(hyper.__dict__)
                pprint(hyper_dict)
                sql_session.add(hyper)
            else:
                hyper = Hyper(x_true, dim)  # new hyper with same parameters
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
    random.shuffle(good_parameters)  # permutes inplace

    for ir in itertools.count():  # repeat the set
        print "ROUND %i" % ir
        print "=========================================="

        for ip, params in enumerate(good_parameters):
            print "parameterset %i" % ip
            print "----------------------------------------------"
            sql_session = Session()

            for percent in [0.25, 0.5, 1]:
                print "percentage of data %.2f" % percent
                print "- - - - - -  - - - - -  - - - -  - - -  - - - "
                new_length = int(len(original_Z) * percent)
                Z = original_Z[:new_length]

                for n_normflows in [1, 2, 3, 4, 8, 20]:
                    print "n_normflows %i" % n_normflows
                    print ".............................................."
                    for ia in xrange(2): # there is a lot randomness involved here
                        print "attempt %i" % ia
                        print ". . . . . . . . . . . . . . . . . . . . . ."
                        hyper = Hyper(x_true, dim)
                        get_init_random(hyper)
                        hyper_init_dict(hyper, params)
                        hyper.n_normflows = n_normflows
                        hyper.percent = percent

                        sql_session.add(hyper)
                        optimize_all(hyper)