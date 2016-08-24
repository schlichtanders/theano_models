# coding: utf-8
from __future__ import division

import contextlib
import gzip
import os, platform, sys, traceback
from functools import partial
from pprint import pformat, pprint

import cPickle
import numpy as np

from breze.learn.base import cast_array_to_local_type
from breze.learn.data import one_hot
from schlichtanders.myplot import add_point
from theano.gof import MethodNotDefined

from climin.util import optimizer
from itertools import repeat, cycle, islice, izip, imap
import random

from schlichtanders.mycontextmanagers import ignored
from schlichtanders.myfunctools import compose, meanmap, summap, compose_fmap, Average, lift
from schlichtanders.mygenerators import eatN, chunk, chunk_list, every, takeN, cycle_permute
from schlichtanders.myobjects import NestedNamespace, Namespace

import theano_models as tm
import theano_models.deterministic_models as dm
import theano_models.probabilistic_models as pm

from sklearn import cross_validation
from theano.tensor.shared_randomstreams import RandomStreams
import theano

from sqlalchemy import Column, Integer, Unicode, UnicodeText, String, PickleType, Float, Boolean
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from copy import copy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
inf = float("inf")
EPS = 1e-4

__file__ = os.path.realpath(__file__)
if platform.system() == "Windows":
    from schlichtanders.myos import replace_unc
    __file__ = replace_unc(__file__)
__path__ = os.path.dirname(__file__)
__parent__ = os.path.dirname(__path__)

# GENERAL HELPERS
# ===============

class Track(object):
    def __getattr__(self, item):
        return tm.track_model(getattr(tm, item))
track = Track()

@contextlib.contextmanager
def log_exceptions(abs_path_errors, title, extra_dict, *exceptions):
    if not exceptions:
        exceptions = Exception
    try:
        yield
    except exceptions:
        with open(abs_path_errors, "a") as myfile:  # os.path.join(__path__, foldername, '%s_errors.txt' % filename)
            error = """
%s
------------
EXTRA INFO: %s
ORIGINAL ERROR: %s""" % (title, pformat(extra_dict), traceback.format_exc())
            myfile.write(error)


def fix_prefix(prefix):
    if prefix and not prefix.endswith("_"):  # source of bugs
        prefix += "_"
    return prefix


# DATA
# ====

def RMSE(PX, Z):
    return np.sqrt(((PX - Z) ** 2).mean())

def error_rate_categorical(PX, Z):
    return (PX[:, :10].argmax(1) != Z.argmax(1)).mean()

def load_and_preprocess_data(datasetname):
    if datasetname.lower() == "mnist":
        datafile = os.path.join(__parent__, 'data', 'mnist.pkl.gz')
        with gzip.open(datafile, 'rb') as f:
            train_set, val_set, test_set = cPickle.load(f)

        # X data is already normalized to (0,1)
        X, Z = train_set
        VX, VZ = val_set
        TX, TZ = test_set

        Z = one_hot(Z, 10)
        VZ = one_hot(VZ, 10)
        TZ = one_hot(TZ, 10)

        data = [cast_array_to_local_type(i) for i in (X, Z, VX, VZ, TX, TZ)]
        return data, error_rate_categorical
    else:
        Z, X = getattr(tm.data, "_" + datasetname)()
        # normalization is standard in Probabilistic Backpropagation Paper
        X_mean = X.mean(0)
        X_std = X.std(0)
        X = (X - X_mean) / X_std
        Z_mean = Z.mean(0)
        Z_std = Z.std(0)
        Z = (Z - Z_mean) / Z_std

        X, TX, Z, TZ = cross_validation.train_test_split(X, Z, test_size=0.1) # 10% test used in paper
        X, VX, Z, VZ = cross_validation.train_test_split(X, Z, test_size=0.1) # 20% validation used in paper
        data = X, Z, VX, VZ, TX, TZ

        def nRMSE(PX, Z):
            return RMSE(PX*Z_std + Z_mean, Z*Z_std + Z_mean)

        return data, nRMSE


# SQLALCHEMY
# ==========
Base = declarative_base()

class Hyper(Base):
    __tablename__ = "hyper"
    id = Column(Integer, primary_key=True)

    # hyper parameters:
    datasetname = Column(String)
    modelname = Column(String)
    optimization_type = Column(String)
    percent = Column(Float)
    max_epochs_without_improvement = Column(Integer)
    logP_average_n_intermediate = Column(Integer)
    errorrate_average_n = Column(Integer)
    units_per_layer = Column(Integer)
    units_per_layer_plus = Column(Integer)
    n_layers = Column(Integer)
    minus_log_s1 = Column(Integer)
    minus_log_s2 = Column(Integer)
    batch_size = Column(Integer)
    annealing_T = Column(Integer, nullable=True)
    adapt_prior = Column(Boolean)

    n_normflows = Column(Integer)

    opt_identifier = Column(String)
    opt_momentum = Column(Float)
    opt_offset = Column(Float)
    opt_decay = Column(Float)
    opt_step_rate = Column(Float)

    best_parameters = Column(PickleType, nullable=True)
    best_val_loss = Column(Float)
    best_test_loss = Column(Float)
    train_loss = Column(PickleType)
    val_loss = Column(PickleType)
    best_epoch = Column(Integer)
    init_parameters = Column(PickleType, nullable=True)
    best_val_error = Column(Float)
    best_test_error = Column(Float)

    example_input = Column(PickleType, nullable=True)
    example_output = Column(PickleType, nullable=True)
    output_transfer = Column(String, nullable=True)

    def __init__(self, datasetname, modelname, optimization_type):
        """
        Parameters
        ----------
        datasetname : str
        """
        self.output_transfer = "identity"  # currently this is always the case
        self.datasetname = datasetname
        self.modelname = modelname
        self.optimization_type = optimization_type
        self.percent = 1.0
        self.max_epochs_without_improvement = 40  # 30
        self.logP_average_n_intermediate = 3  # TODO random.choice([1,10])
        self.logP_average_n_final = 10
        self.errorrate_average_n = 10
        self.output_transfer = "identity"
        self.annealing_T = 100  # in terms of epochs; 50 may be good
        self.adapt_prior = False
        self.init_parameters = None  # half result parameter, half hyperparameter, but more hyperparameter
        self.init_results()

    def init_results(self):
        self.best_parameters = None
        self.best_val_loss = inf
        self.best_test_loss = inf
        self.train_loss = []
        self.val_loss = []
        self.best_epoch = 0
        self.best_val_error = inf
        self.best_test_error = inf

    def __repr__(self):
        return "hyper %i" % hash(self)

def get_init_data(data):
    hyper = Namespace()
    X, Z = data[:2]
    hyper.example_input, hyper.example_output = X[0], Z[0]
    return hyper.__dict__

def hyper_init_dict(hyper, hyper_dict, prefix="", do_copy=True):
    prefix = fix_prefix(prefix)
    for k, v in hyper_dict.iteritems():
        if not k.startswith("_"):
            setattr(hyper, prefix + k, copy(v) if do_copy else v)
    return hyper


def get_init_random():  # we directly refer to dict as sqlalchemy deletes the dict once committed (probably for detecting changes
    # batch_size=2 for comparison with maximum-likelihood (dimensions error was thrown in exactly those cases for batch_size=1
    # there are still erros with batch_size=2 for some weird reasons... don't know. I hope this is not crucial.
    hyper = Namespace()
    hyper.batch_size = random.choice([1, 10, 50, 100])
    hyper.minus_log_s1 = random.choice([1, 2, 3, 4, 5, 6, 7])
    hyper.minus_log_s2 = random.choice([6, 7, 8])
    # the prior is learned together with the other models in analogy to the paper Probabilistic Backpropagation

    hyper.n_normflows = random.choice([1, 2, 3, 4, 8, 20])  # 32 is to much for theano... unfortunately

    hyper.opt_identifier = random.choice(["adadelta", "adam", "rmsprop"])
    if hyper.opt_identifier == "adadelta":
        hyper.opt_momentum = random.choice([np.random.uniform(0, 0.01), np.random.uniform(0.9, 1)])
        # self.opt_offset = random.choice([5e-5, 1e-8])
    elif hyper.opt_identifier == "adam":
        hyper.opt_momentum = random.choice([np.random.uniform(0, 0.01), np.random.uniform(0.8, 0.93)])
        # self.opt_offset = 10 ** -np.random.uniform(3, 4)
    elif hyper.opt_identifier == "rmsprop":
        hyper.opt_momentum = random.choice([np.random.uniform(0.002, 0.008), np.random.uniform(0.9, 1)])
        # self.opt_offset = np.random.uniform(0, 0.000045)
    # self.opt_step_rate = random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    hyper.opt_step_rate = random.choice([1e-3, 1e-4, 1e-5, 1e-6])
    hyper.opt_decay = np.random.uniform(0.78, 1)
    return hyper.__dict__


def get_init_several():
    hyper = Namespace()
    hyper.max_epochs_without_improvement = 40  # 30
    # batch_size=2 for comparison with maximum-likelihood (dimensions error was thrown in exactly those cases for batch_size=1
    # there are still erros with batch_size=2 for some weird reasons... don't know. I hope this is not crucial.
    hyper.n_layers = 1  # random.choice([1, 2])  # for comparison of parameters, n_layers=1 has crucial advantages
    hyper.units_per_layer = 50  # random.choice([50, 200]) - check whether baselineplus has an effect
    # the following formula is sound for n_layers=1
    # try:
    #     hyper.units_per_layer_plus = hyper.units_per_layer + hyper.units_per_layer * (hyper.n_normflows * 2)
    # except TypeError:
    #     # n_normflows not yet set
    warnings.warn("don't forget to set hyper.units_per_layer_plus")
    return hyper.__dict__

def get_init_mnist():
    hyper = Namespace()
    hyper.max_epochs_without_improvement = 5  # 30 epochs need extremely long here
    # batch_size=2 for comparison with maximum-likelihood (dimensions error was thrown in exactly those cases for batch_size=1
    # there are still erros with batch_size=2 for some weird reasons... don't know. I hope this is not crucial.
    hyper.batch_size = 128
    hyper.n_layers = 2  # random.choice([1, 2])  # for comparison of parameters, n_layers=1 has crucial advantages
    hyper.units_per_layer = 400  # random.choice([50, 200]) - check whether baselineplus has an effect
    hyper.units_per_layer_plus = 1200
    return hyper.__dict__


# OPTIMIZATION
# ============

def standard_flat(model):
    all_params = []
    if 'parameters_positive' in model:
        # all_params += tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
        all_params += tm.prox_reparameterize(model['parameters_positive'], track.squareplus, track.squareplus_inv)
    if "parameters_psumto1" in model:
        all_params += tm.prox_reparameterize(model['parameters_psumto1'], tm.softmax, tm.softmax_inv)
    all_params += model['parameters']
    return tm.prox_flatten(tm.prox_center(all_params))


def optimize(data, hyper, model, error_func, plot_val=None, plot_best_val=None):
    # choose right loss
    if "ml" in hyper.optimization_type:
        loss = tm.loss_probabilistic(model)
    elif "annealing" in hyper.optimization_type:
        if hyper.annealing_T is None:
            loss = tm.loss_variational(model)
        else:
            loss = tm.loss_normalizingflow(model)
    parameters = standard_flat(model)

    hyper.init_results()  # delete everything which might be there from another run
    X, Z, VX, VZ, TX, TZ = data[:6]
    val_args = (VZ,) if VX is None else (VZ, VX)
    test_args = (TZ,) if TX is None else (TZ, TX)
    val_kwargs = {}
    if "annealing" in hyper.optimization_type:
        val_kwargs['no_annealing'] = True
    tm.reduce_all_identities()

    n_batches = Z.shape[0] // hyper.batch_size  # after this many steps we went through the whole data set once
    mycycle = cycle_permute if hyper.annealing_T is None and hyper.optimization_type == "annealing" else cycle
    if X is None:
        climin_args = izip(imap(lambda z: (z,), chunk(hyper.batch_size, mycycle(Z))), repeat({}))
    else:
        climin_args = izip(izip(chunk(hyper.batch_size, mycycle(Z)), chunk(hyper.batch_size, mycycle(X))), repeat({}))

    if hyper.optimization_type == "ml":  # maximum likelihood
        numericalize_kwargs = dict(
            batch_mapreduce=meanmap,
        )
    elif hyper.optimization_type == "ml_exp_average":
        numericalize_kwargs = dict(
            batch_mapreduce=meanmap,
            exp_average_n=hyper.exp_average_n,
            exp_ratio_estimator=hyper.exp_ratio_estimator,
        )
    elif hyper.optimization_type == "annealing":
        if hyper.annealing_T is not None:
            def weights_data():
                T = n_batches * hyper.annealing_T
                for t in xrange(T):
                    yield min(1, 0.01 + t / T)  # 10000
                while True:
                    yield 1

            numericalize_kwargs = dict(
                batch_mapreduce=summap,  # meaning is/must be done in Annealing
                annealing_combiner=tm.AnnealingCombiner(
                    weights_data=weights_data()
                ),
            )
        else:
            def weights_regularizer_1epoch():
                for i in range(1, n_batches + 1):
                    yield 2 ** (n_batches - i) / (2 ** n_batches - 1)

            assert len(list(weights_regularizer_1epoch())) == n_batches
            numericalize_kwargs = dict(
                batch_mapreduce=summap,  # meaning is/must be done in Annealing
                annealing_combiner=tm.AnnealingCombiner(
                    weights_regularizer=cycle(weights_regularizer_1epoch())
                ),
            )
        if hyper.batch_size == 1:
            numericalize_kwargs['batch_precompile'] = "singleton"
    else:
        raise ValueError("Unkown type %s" % hyper.optimization_type)

    def _optimize(mode='FAST_RUN'):
        theano.config.mode = mode
        def adapt_init_params(ps):
            if hyper.init_parameters is None:
                print "new random init_parameters"
                return ps + np.random.normal(size=ps.shape, scale=1) # better more initial randomness
            else:
                print "used given init_parameters"
                return hyper.init_parameters

        optimizer_kwargs = tm.numericalize(
            loss, parameters,
            adapt_init_params=adapt_init_params,
            # profile=True,
            mode=mode,
            **numericalize_kwargs
        )

        opt = optimizer(
            identifier=hyper.opt_identifier,
            step_rate=hyper.opt_step_rate,
            momentum=hyper.opt_momentum,
            decay=hyper.opt_decay,
            # offset=hyper.opt_offset,
            args=climin_args,
            **tm.climin_kwargs(optimizer_kwargs)
        )
        # start values:
        hyper.init_parameters = copy(opt.wrt)
        hyper.best_val_loss = optimizer_kwargs['num_loss'](opt.wrt, *val_args, **val_kwargs)
        if plot_val is not None:
            add_point(plot_val, 0, hyper.best_val_loss)
        if plot_best_val is not None:
            add_point(plot_best_val, 0, hyper.best_val_loss)
        # for the start no averaging is needed, as this is not crucial at all

        hyper.val_loss = []
        # train_losses = getattr(hyper.train_loss")
        for info in every(n_batches, opt):
            current_epoch = info['n_iter'] // n_batches
            print current_epoch,
            if current_epoch - hyper.best_epoch > hyper.max_epochs_without_improvement:
                break
            # collect and visualize validation loss for choosing the best model
            # val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, **val_kwargs)
            if hyper.logP_average_n_intermediate <= 1 or hyper.optimization_type.startswith("ml"):  # maximum_likelihood already averages over each single data point
                val_loss = optimizer_kwargs['num_loss'](opt.wrt, *val_args, **val_kwargs)
            else:  # as we use batch_common_rng = True by default, for better comparison, average over several noisy weights:
                val_loss = Average(hyper.logP_average_n_intermediate)(
                    optimizer_kwargs['num_loss'], opt.wrt, *val_args, **val_kwargs)
            if val_loss < hyper.best_val_loss - EPS:
                hyper.best_epoch = current_epoch
                hyper.best_parameters = copy(opt.wrt)  # copy is needed as climin works inplace on array
                hyper.best_val_loss = val_loss
                if plot_best_val is not None:
                    add_point(plot_best_val, current_epoch, val_loss)
            hyper.val_loss.append(val_loss)
            if plot_val is not None:
                add_point(plot_val, current_epoch, val_loss)

            # visualize training loss for comparison:
            # training_loss = optimizer_kwargs['num_loss'](opt.wrt, Z[:10], X[:10], no_annealing=True)
            # train_losses.append(training_loss)
        print
        if hyper.best_parameters is not None:
            hyper.best_val_loss = Average(hyper.logP_average_n_final)(
                optimizer_kwargs['num_loss'], hyper.best_parameters, *val_args, **val_kwargs)
            hyper.best_test_loss = Average(hyper.logP_average_n_final)(
                optimizer_kwargs['num_loss'], hyper.best_parameters, *test_args, **val_kwargs)

    try:
        _optimize()
    except MethodNotDefined:  # this always refers to the limit of 32 nodes per ufunc... weird issue
        _optimize(mode='FAST_COMPILE')

    if hyper.best_parameters is not None:

        # there problems with down_casting 64bit to 32bit float vectors... I cannot see where the point is, however
        # the version below works indeed
        # predict = model.function(givens={parameters: getattr(hyper, prefix + "best_parameters")}, allow_input_downcast=True)
        # predict = lift(predict, Average(hyper.errorrate_average_n))
        # PVX = np.apply_along_axis(predict, 1, VX)
        # setattr(hyper, prefix + 'best_val_error', error_func(PVX, VZ))

        # test error rate:
        fmap_avg = Average(hyper.errorrate_average_n)
        sampler = theano.function([parameters] + model['inputs'], model['outputs'], allow_input_downcast=True)

        PVX = []
        for i in xrange(len(VZ)):
            val_sample_args = (VX[i],) if VX is not None else tuple()
            PVX.append(fmap_avg(sampler, hyper.best_parameters, *val_sample_args))
        PVX = np.array(PVX)

        PTX = []
        for i in xrange(len(TZ)):
            test_sample_args = (TX[i],) if TX is not None else tuple()
            PTX.append(fmap_avg(sampler, hyper.best_parameters, *test_sample_args))
        PTX = np.array(PTX)
        hyper.best_val_error = error_func(PVX, VZ)
        hyper.best_test_error = error_func(PTX, TZ)

    return hyper
