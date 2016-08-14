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

from experiment_util import log_exceptions, setup_sqlite, hyper_init_dict, hyper_init_random, \
    optimize, load_and_preprocess_data, hyper_init_several, hyper_init_mnist, RMSE
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
#overwrite as far as given:
if len(sys.argv) > 2:
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
}
model_prefixes = reduce(op.add, model_names.values())
model_prefixes = [p+"_" for p in model_prefixes]

# Hyperparameters
# ===============

Base = declarative_base()

class Hyper(Base):
    __tablename__ = "hyper"
    id = Column(Integer, primary_key=True)

    # hyper parameters:
    x_true = Column(Float)
    max_epochs_without_improvement = Column(Integer)
    logP_average_n = Column(Integer)
    errorrate_average_n = Column(Integer)
    minus_log_s1 = Column(Integer)
    minus_log_s2 = Column(Integer)
    batch_size = Column(Integer)

    n_normflows = Column(Integer)

    opt_identifier = Column(String)
    opt_momentum = Column(Float)
    opt_offset = Column(Float)
    opt_decay = Column(Float)
    opt_step_rate = Column(Float)

    for _prefix in model_prefixes:
        exec("""
{0}best_val_loss = Column(Float)
{0}best_parameters = Column(PickleType, nullable=True)
{0}train_loss = Column(PickleType)
{0}val_loss = Column(PickleType)
{0}epochs = Column(Integer)
{0}init_params = Column(PickleType, nullable=True)
{0}val_error_rate = Column(Float)""".format(_prefix))
    def __init__(self, x_true):
        """
        Parameters
        ----------
        datasetname : str
        """
        self.x_true = x_true
        self.max_epochs_without_improvement = 30
        self.logP_average_n = 3  # TODO random.choice([1,10])
        self.errorrate_average_n = 10
        self.init_results()

    def init_results(self):

        # extra for being able to reset results for loaded hyperparameters
        for prefix in model_prefixes:
            setattr(self, prefix + "best_parameters", None)
            setattr(self, prefix + "best_val_loss", inf)
            setattr(self, prefix + "train_loss", [])
            setattr(self, prefix + "val_loss", [])
            setattr(self, prefix + "best_epoch", 0)
            setattr(self, prefix + "init_params", None)
            setattr(self, prefix + "val_error_rate", inf)

engine = create_engine('sqlite:///' + filepath)  # os.path.join(__path__, foldername, '%s.db' % filename)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
sql_session = Session()





# DATA
# ====

# pm.RNG = NestedNamespace(tm.PooledRandomStreams(pool_size=int(1e8)), RandomStreams())

# The function:
sampler = experiment_toy_models.toy_likelihood().function()
x_true = 0.65
_x_true = np.array([x_true], dtype=theano.config.floatX)
Z = np.array([sampler(_x_true) for n in range(1000)], dtype=theano.config.floatX)
Z, TZ = cross_validation.train_test_split(Z, test_size=0.1)  # 10% test used in paper
Z, VZ = cross_validation.train_test_split(Z, test_size=0.1)  # 20% validation used in paper
data = None, Z, None, VZ, None, TZ  # None represents X data
error_func = RMSE

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
                hyper = Hyper(x_true)
                hyper_init_random(hyper)
                hyper_dict = copy(hyper.__dict__)
                pprint(hyper_dict)
                sql_session.add(hyper)
            else:
                hyper = Hyper(x_true)  # new hyper with same parameters
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
            for n_normflows in [1, 2, 3, 8]:
                print "n_normflows %i" % n_normflows
                print ".............................................."
                for ia in xrange(3): # there is a lot randomness involved here
                    print "attempt %i" % ia
                    print ". . . . . . . . . . . . . . . . . . . . . ."
                    hyper = Hyper(x_true)
                    hyper_init_random(hyper)
                    hyper_init_dict(hyper, params)
                    hyper.n_normflows = n_normflows

                    sql_session.add(hyper)
                    optimize_all(hyper)