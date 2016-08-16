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
from theano.gof import MethodNotDefined

from climin.util import optimizer
from itertools import repeat, cycle, islice, izip, imap
import random

from schlichtanders.mycontextmanagers import ignored
from schlichtanders.myfunctools import compose, meanmap, summap, compose_fmap, Average, lift
from schlichtanders.mygenerators import eatN, chunk, chunk_list, every, takeN
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
def get_hyper(model_prefixes):
    model_prefixes = map(fix_prefix, model_prefixes)

    Base = declarative_base()

    class Hyper(Base):
        __tablename__ = "hyper"
        id = Column(Integer, primary_key=True)

        # hyper parameters:
        datasetname = Column(String)
        max_epochs_without_improvement = Column(Integer)
        logP_average_n = Column(Integer)
        errorrate_average_n = Column(Integer)
        exp_average_n = Column(Integer)
        exp_ratio_estimator = Column(String, nullable=True)
        units_per_layer = Column(Integer)
        units_per_layer_plus = Column(Integer)
        n_layers = Column(Integer)
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
            exec ("""
{0}best_parameters = Column(PickleType, nullable=True)
{0}best_val_loss = Column(Float)
{0}best_test_loss = Column(Float)
{0}train_loss = Column(PickleType)
{0}val_loss = Column(PickleType)
{0}epochs = Column(Integer)
{0}init_params = Column(PickleType, nullable=True)
{0}val_error_rate = Column(Float)
{0}test_error_rate = Column(Float)""".format(_prefix))

        def __init__(self, datasetname):
            """
            Parameters
            ----------
            datasetname : str
            """
            self.datasetname = datasetname
            self.max_epochs_without_improvement = 30
            self.logP_average_n = 3  # TODO random.choice([1,10])
            self.errorrate_average_n = 10
            self.exp_average_n = 20
            self.init_results()

        def init_results(self):

            # extra for being able to reset results for loaded hyperparameters
            for prefix in model_prefixes:
                setattr(self, prefix + "best_parameters", None)
                setattr(self, prefix + "best_val_loss", inf)
                setattr(self, prefix + "best_test_loss", inf)
                setattr(self, prefix + "train_loss", [])
                setattr(self, prefix + "val_loss", [])
                setattr(self, prefix + "best_epoch", 0)
                setattr(self, prefix + "init_params", None)
                setattr(self, prefix + "val_error_rate", inf)
                setattr(self, prefix + "test_error_rate", inf)

        def __repr__(self):
            return "hyper %i" % hash(self)

    return Hyper

def get_old_hyper(model_prefixes):
    model_prefixes = map(fix_prefix, model_prefixes)

    Base = declarative_base()

    class Hyper(Base):
        __tablename__ = "hyper"
        id = Column(Integer, primary_key=True)

        # hyper parameters:
        datasetname = Column(String)
        max_epochs_without_improvement = Column(Integer)
        logP_average_n = Column(Integer)
        errorrate_average_n = Column(Integer)
        exp_average_n = Column(Integer)
        exp_ratio_estimator = Column(String, nullable=True)
        units_per_layer = Column(Integer)
        units_per_layer_plus = Column(Integer)
        n_layers = Column(Integer)
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
            exec ("""
{0}best_val_loss = Column(Float)
{0}best_parameters = Column(PickleType, nullable=True)
{0}train_loss = Column(PickleType)
{0}val_loss = Column(PickleType)
{0}epochs = Column(Integer)
{0}init_params = Column(PickleType, nullable=True)
{0}val_error_rate = Column(Float)""".format(_prefix))

        def __init__(self, datasetname):
            """
            Parameters
            ----------
            datasetname : str
            """
            self.datasetname = datasetname
            self.max_epochs_without_improvement = 30
            self.logP_average_n = 3  # TODO random.choice([1,10])
            self.errorrate_average_n = 10
            self.exp_average_n = 20
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

        def __repr__(self):
            return "hyper %i" % hash(self)

    return Hyper

def setup_sqlite(model_prefixes, abs_path_sqlite):
    """
    class factory

    Parameters
    ----------
    model_prefixes : list of str
    abs_path_sqlite : str

    Returns
    -------
    Hyper, session
    Hyperparameter class for use qith sqlalchemy
    """
    Hyper = get_hyper(model_prefixes)
    engine = create_engine('sqlite:///' + abs_path_sqlite)  # os.path.join(__path__, foldername, '%s.db' % filename)
    Hyper.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Hyper, Session()


def hyper_init_dict(hyper, hyper_dict, prefix="", do_copy=True):
    prefix = fix_prefix(prefix)
    for k, v in hyper_dict.iteritems():
        if not k.startswith("_"):
            setattr(hyper, prefix + k, copy(v) if do_copy else v)
    return hyper


def hyper_init_random(hyper):  # we directly refer to dict as sqlalchemy deletes the dict once committed (probably for detecting changes
    # batch_size=2 for comparison with maximum-likelihood (dimensions error was thrown in exactly those cases for batch_size=1
    # there are still erros with batch_size=2 for some weird reasons... don't know. I hope this is not crucial.
    hyper.batch_size = random.choice([1, 10, 50, 100])
    hyper.exp_ratio_estimator = random.choice([None, "grouping", "firstorder"])
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


def hyper_init_several(hyper):
    hyper.max_epochs_without_improvement = 30
    # batch_size=2 for comparison with maximum-likelihood (dimensions error was thrown in exactly those cases for batch_size=1
    # there are still erros with batch_size=2 for some weird reasons... don't know. I hope this is not crucial.
    hyper.n_layers = 1  # random.choice([1, 2])  # for comparison of parameters, n_layers=1 has crucial advantages
    hyper.units_per_layer = 50  # random.choice([50, 200]) - check whether baselineplus has an effect
    # the following formula is sound for n_layers=1
    try:
        hyper.units_per_layer_plus = hyper.units_per_layer + hyper.units_per_layer * (hyper.n_normflows * 2)
    except TypeError:
        # n_normflows not yet set
        warnings.warn("don't forget to set hyper.units_per_layer_plus")

def hyper_init_mnist(hyper):
    hyper.max_epochs_without_improvement = 5  # 30 epochs need extremely long here
    # batch_size=2 for comparison with maximum-likelihood (dimensions error was thrown in exactly those cases for batch_size=1
    # there are still erros with batch_size=2 for some weird reasons... don't know. I hope this is not crucial.
    hyper.batch_size = 128
    hyper.n_layers = 2  # random.choice([1, 2])  # for comparison of parameters, n_layers=1 has crucial advantages
    hyper.units_per_layer = 400  # random.choice([50, 200]) - check whether baselineplus has an effect
    hyper.units_per_layer_plus = 1200


# OPTIMIZATION
# ============

def optimize(prefix, data, hyper, model, loss, parameters, error_func, optimization_type):
    X, Z, VX, VZ, TX, TZ = data[:6]
    print prefix
    if prefix and not prefix.endswith("_"):  # source of bugs
        prefix += "_"
    tm.reduce_all_identities()

    n_batches = Z.shape[0] // hyper.batch_size  # after this many steps we went through the whole data set once
    if X is None:
        climin_args = izip(imap(lambda x: (x,), chunk(hyper.batch_size, cycle(Z))), repeat({}))
    else:
        climin_args = izip(izip(chunk(hyper.batch_size, cycle(Z)), chunk(hyper.batch_size, cycle(X))), repeat({}))

    if optimization_type == "ml": # maximum likelihood
        numericalize_kwargs = dict(
            batch_mapreduce=meanmap,
        )
    elif optimization_type == "ml_exp_average":
        numericalize_kwargs = dict(
            batch_mapreduce=meanmap,
            exp_average_n=hyper.exp_average_n,
            exp_ratio_estimator=hyper.exp_ratio_estimator,
        )
    elif optimization_type == "annealing":
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
    else:
        raise ValueError("Unkown type %s" % optimization_type)

    def _optimize(mode='FAST_RUN'):
        theano.config.mode = mode
        optimizer_kwargs = tm.numericalize(
            loss, parameters,
            adapt_init_params=lambda ps: ps + np.random.normal(size=ps.size, scale=1), # better more initial randomness
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

        val_kwargs = {}
        if optimization_type == "annealing":
            val_kwargs['no_annealing'] = True
        # start values:
        setattr(hyper, prefix + "init_params", copy(opt.wrt))
        if VX is None:
            setattr(hyper, prefix + "best_val_loss", optimizer_kwargs['num_loss'](opt.wrt, VZ, **val_kwargs))
        else:
            setattr(hyper, prefix + "best_val_loss", optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, **val_kwargs))
        # for the start no averaging is needed, as this is not crucial at all

        setattr(hyper, prefix + "val_loss", [])
        val_losses = getattr(hyper, prefix + "val_loss")
        # train_losses = getattr(hyper, prefix + "train_loss")
        for info in every(n_batches, opt):
            current_epoch = info['n_iter'] // n_batches
            print current_epoch,
            if current_epoch - getattr(hyper, prefix + "best_epoch") > hyper.max_epochs_without_improvement:
                break
            # collect and visualize validation loss for choosing the best model
            # val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, **val_kwargs)
            if hyper.logP_average_n <= 1 or optimization_type.startswith("ml"):  # maximum_likelihood already averages over each single data point
                if VX is None:
                    val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, **val_kwargs)
                else:
                    val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, **val_kwargs)
            else:  # as we use batch_common_rng = True by default, for better comparison, average over several noisy weights:
                if VX is None:
                    val_loss = Average(hyper.logP_average_n)(optimizer_kwargs['num_loss'], opt.wrt, VZ, **val_kwargs)
                else:
                    val_loss = Average(hyper.logP_average_n)(optimizer_kwargs['num_loss'], opt.wrt, VZ, VX, **val_kwargs)
            if val_loss < getattr(hyper, prefix + "best_val_loss") - EPS:
                setattr(hyper, prefix + "best_epoch", current_epoch)
                setattr(hyper, prefix + "best_parameters", copy(opt.wrt))  # copy is needed as climin works inplace on array
                setattr(hyper, prefix + "best_val_loss", val_loss)
            val_losses.append(val_loss)

            # visualize training loss for comparison:
            # training_loss = optimizer_kwargs['num_loss'](opt.wrt, Z[:10], X[:10], no_annealing=True)
            # train_losses.append(training_loss)
        print
        best_params = getattr(hyper, prefix + "best_parameters")
        if best_params is not None:
            if TX is None:
                setattr(hyper, prefix + "best_test_loss", optimizer_kwargs['num_loss'](best_params, TZ, **val_kwargs))
            else:
                setattr(hyper, prefix + "best_test_loss", optimizer_kwargs['num_loss'](best_params, TZ, TX, **val_kwargs))

    try:
        _optimize()
    except MethodNotDefined:  # this always refers to the limit of 32 nodes per ufunc... weird issue
        _optimize(mode='FAST_COMPILE')

    if getattr(hyper, prefix + "best_parameters") is not None:

        # there problems with down_casting 64bit to 32bit float vectors... I cannot see where the point is, however
        # the version below works indeed
        # predict = model.function(givens={parameters: getattr(hyper, prefix + "best_parameters")}, allow_input_downcast=True)
        # predict = lift(predict, Average(hyper.errorrate_average_n))
        # PVX = np.apply_along_axis(predict, 1, VX)
        # setattr(hyper, prefix + 'val_error_rate', error_func(PVX, VZ))

        # test error rate:
        if VX is not None or TX is not None:
            best_params = getattr(hyper, prefix + "best_parameters")
            fmap_avg = Average(hyper.errorrate_average_n)
            def avg_predict(x):
                return fmap_avg(sampler, best_params, x)
            sampler = theano.function([parameters] + model['inputs'], model['outputs'], allow_input_downcast=True)

        if VX is not None:  # sometimes the above does not even run one epoch
            PVX = np.apply_along_axis(avg_predict, 1, VX)
            setattr(hyper, prefix + 'val_error_rate', error_func(PVX, VZ))
        if TX is not None:
            PTX = np.apply_along_axis(avg_predict, 1, TX)
            setattr(hyper, prefix + 'test_error_rate', error_func(PTX, TZ))

    return hyper


def test(data, hyper, model, loss, parameters, error_func, optimization_type, init_params):
    if hasattr(hyper, "datasetname"):
        print hyper.datasetname
    X, Z, VX, VZ, TX, TZ = data
    tm.reduce_all_identities()

    n_batches = Z.shape[0] // hyper.batch_size  # after this many steps we went through the whole data set once
    if X is None:
        climin_args = izip(imap(lambda x: (x,), chunk(hyper.batch_size, cycle(Z))), repeat({}))
    else:
        climin_args = izip(izip(chunk(hyper.batch_size, cycle(Z)), chunk(hyper.batch_size, cycle(X))), repeat({}))

    if optimization_type == "ml":  # maximum likelihood
        numericalize_kwargs = dict(
            batch_mapreduce=meanmap,
        )
    elif optimization_type == "ml_exp_average":
        numericalize_kwargs = dict(
            batch_mapreduce=meanmap,
            exp_average_n=hyper.exp_average_n,
            exp_ratio_estimator=hyper.exp_ratio_estimator,
        )
    elif optimization_type == "annealing":
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
    else:
        raise ValueError("Unkown type %s" % optimization_type)

    def _optimize(mode='FAST_RUN'):
        theano.config.mode = mode
        optimizer_kwargs = tm.numericalize(
            loss, parameters,
            adapt_init_params=lambda ps: init_params,
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

        test_kwargs = {}
        if optimization_type == "annealing":
            test_kwargs['no_annealing'] = True
        if TX is None:
            best_test_loss = optimizer_kwargs['num_loss'](opt.wrt, TZ, **test_kwargs)
        else:
            best_test_loss = optimizer_kwargs['num_loss'](opt.wrt, TZ, TX, **test_kwargs)
        best_epoch = 0
        best_parameters = None
        # for the start no averaging is needed, as this is not crucial at all
        # train_losses = getattr(hyper, prefix + "train_loss")
        for info in every(n_batches, opt):
            current_epoch = info['n_iter'] // n_batches
            print current_epoch,
            if current_epoch - best_epoch > hyper.max_epochs_without_improvement:
                break
            # collect and visualize validation loss for choosing the best model
            # val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, **val_kwargs)
            if hyper.logP_average_n <= 1 or optimization_type.startswith("ml"):  # maximum_likelihood already averages over each single data point
                if TX is None:
                    test_loss = optimizer_kwargs['num_loss'](opt.wrt, TZ, **test_kwargs)
                else:
                    test_loss = optimizer_kwargs['num_loss'](opt.wrt, TZ, TX, **test_kwargs)
            else:  # as we use batch_common_rng = True by default, for better comparison, average over several noisy weights:
                if TX is None:
                    test_loss = Average(hyper.logP_average_n)(optimizer_kwargs['num_loss'], opt.wrt, TZ, **test_kwargs)
                else:
                    test_loss = Average(hyper.logP_average_n)(optimizer_kwargs['num_loss'], opt.wrt, TZ, TX, **test_kwargs)
            if test_loss < best_test_loss - EPS:
                best_epoch = current_epoch
                best_test_loss = test_loss
                best_parameters = copy(opt.wrt)

            # visualize training loss for comparison:
            # training_loss = optimizer_kwargs['num_loss'](opt.wrt, Z[:10], X[:10], no_annealing=True)
            # train_losses.append(training_loss)
        print
        return best_test_loss, best_epoch, best_parameters

    try:
        best_test_loss, best_epoch, best_params = _optimize()
    except MethodNotDefined:  # this always refers to the limit of 32 nodes per ufunc... weird issue
        best_test_loss, best_epoch, best_params = _optimize(mode='FAST_COMPILE')

    test_error_rate = inf
    if best_params is not None and TX is not None:  # sometimes the above does not even run one epoch

        # there problems with down_casting 64bit to 32bit float vectors... I cannot see where the point is, however
        # the version below works indeed
        # predict = model.function(givens={parameters: getattr(hyper, prefix + "best_parameters")}, allow_input_downcast=True)
        # predict = lift(predict, Average(hyper.errorrate_average_n))
        # PVX = np.apply_along_axis(predict, 1, VX)
        # setattr(hyper, prefix + 'val_error_rate', error_func(PVX, VZ))

        # test error rate:
        fmap_avg = Average(hyper.errorrate_average_n)
        sampler = theano.function([parameters] + model['inputs'], model['outputs'], allow_input_downcast=True)

        def avg_predict(x):
            return fmap_avg(sampler, best_params, x)

        PTX = np.apply_along_axis(avg_predict, 1, TX)
        test_error_rate = error_func(PTX, TZ)
    return test_error_rate, best_test_loss, best_epoch, best_params
