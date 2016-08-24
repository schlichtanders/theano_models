# coding: utf-8
from __future__ import division

import operator as op
import os, platform, sys
import random
from ast import literal_eval
from pprint import pformat, pprint

import itertools
import numpy as np
import csv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from schlichtanders.mycontextmanagers import ignored
from schlichtanders.mydicts import update
from schlichtanders.myobjects import NestedNamespace, Namespace

import theano_models as tm
import theano_models.deterministic_models as dm
import theano_models.probabilistic_models as pm
from theano.tensor.shared_randomstreams import RandomStreams
from copy import copy
import warnings

from experiment_util import log_exceptions, hyper_init_dict, get_init_random, \
    optimize, load_and_preprocess_data, get_init_several, get_init_mnist, Hyper, get_init_data
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
    foldername, filename = sys.argv[1:3]
elif len(sys.argv) > 1:
    foldername = sys.argv[1]

with ignored(OSError):
    os.mkdir(os.path.join(__path__, foldername))
filepath = os.path.join(__path__, foldername, '%s.db' % filename)
errorfilepath = os.path.join(__path__, foldername, '%s_errors.txt' % filename)
csvpath = os.path.join(__path__, 'good_parameters_betteraspbp.csv')


pm.RNG = NestedNamespace(tm.PooledRandomStreams(pool_size=int(1e8)), RandomStreams())

# # Data
#     # datasetnames = ["boston", "concrete", "energy", "kin8nm", "naval", "powerplant", "protein", "winered", "yacht", "year"]
#     datasetnames = ["boston", "concrete", "energy", "kin8nm", "powerplant", "winered", "yacht"]

# "naval" is not possible as standardization leads to a nan column. Hence we do not know which to use
# datasetname = "concrete"


model_names = { # sorted by optimization type
    "ml": ['baselinedet'],
    "annealing": ['baseline', 'planarflow', 'planarflowdet', 'radialflow', 'radialflowdet'],
    # first trials do not seem to be successfull, furthermore this needs a lot of time, maybe later on
    # "ml_exp_average": ['mixtureml', 'planarflowml', 'radialflowml'],
}

# Hyperparameters
# ===============

engine = create_engine('sqlite:///' + filepath)  # os.path.join(__path__, foldername, '%s.db' % filename)
Hyper.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
sql_session = Session()

get_init_dataset = get_init_mnist if datasetname == "mnist" else get_init_several


# Data
# ====
data, error_func = load_and_preprocess_data(datasetname)
X, Z, VX, VZ, TX, TZ = data
original_X = X
original_Z = Z
example_input = X[0]
example_output = Z[0]
output_transfer = "softmax" if datasetname=="mnist" else "identity"

# TODO the above randomly splits the dataset, which should be averaged out ideally... however with same initial parameters... that makes it difficult

# Main Loop
# =========
def optimize_all(sql_session, hyper_dict):
    data = X, Z, VX, VZ, TX, TZ  # as we change X and Z during the script
    for optimization_type in model_names:
        for modelname in model_names[optimization_type]:
            # reset hard the saved models, to prevent any possible interaction (hard to debug)
            dm.InvertibleModel.INVERTIBLE_MODELS = []
            tm.Model.all_models = []

            with log_exceptions(errorfilepath, modelname, hyper_dict):
                hyper = Hyper(datasetname, modelname, optimization_type)
                hyper_init_dict(hyper, hyper_dict)
                sql_session.add(hyper)

                model, approx_posterior = getattr(experiment_models, modelname)(hyper)
                optimize(  # updates hyper inplace with results
                    data=data, hyper=hyper, model=model, error_func=error_func
                )
                sql_session.commit()


good_parameters = []
with open(csvpath, "r") as f:
    reader = csv.DictReader(f, quoting=csv.QUOTE_NONE)
    for row in reader:
        # this should not be updated in hyper
        # (double quote as quoting=csv.QUOTE_NONNUMERIC was used to create this csv)
        # del row['"datasetname"']
        # TODO if the first runs through, delete also n_normflows, as we want to have all of them
        good_parameters.append(
            {literal_eval(k): literal_eval(v) for k, v in row.iteritems()}
        )  # evaluate everything, this version also distinguishes ints/floats

good_parameters = sorted(good_parameters, key=lambda gp: gp["datasetname"]!=datasetname)  # 0 for datasetname, 1 else, hence datasetname comes first
# random.shuffle(good_parameters)  # permutes inplace
good_parameters_keys = ["batch_size", "minus_log_s1", "opt_decay", "opt_identifier", "opt_momentum", "opt_offset", "opt_step_rate"]

percent = 1.0
for ir in itertools.count():  # repeat the set
    print "ROUND %i" % ir
    print "=========================================="

    for ip, params in enumerate(good_parameters):
        print "parameterset %i" % ip
        print "----------------------------------------------"
        params = {k: params[k] for k in good_parameters_keys}  # take only needed parameter keys and ignore the rest
        update(params, get_init_random(), overwrite=False)
        update(params, get_init_dataset(), overwrite=True)
        update(params, get_init_data(data), overwrite=True)

        # for percent in [0.5, 1]:
        #     print "percentage of data %.2f" % percent
        #     print "- - - - - -  - - - - -  - - - -  - - -  - - - "
        #     new_length = int(len(original_X) * percent)
        #     X = original_X[:new_length]
        #     Z = original_Z[:new_length]
        for nn in [2, 5, 10, 30] if "toy" in datasetname else [10, 30]:
            print "n_normflows %i" % nn
            print ".............................................."
            for ia in xrange(1):  # there is a lot randomness involved here
                print "attempt %i" % ia
                print ". . . . . . . . . . . . . . . . . . . . . ."
                params['n_normflows'] = nn
                params['percent'] = percent
                if datasetname != "mnist":  # we need the n_normflow parameter for this to be valid
                    params['units_per_layer_plus'] = params['units_per_layer'] + params['units_per_layer'] * (params['n_normflows'] * 2)
                optimize_all(Session(), params)
