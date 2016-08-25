#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os, platform, sys
import warnings
from sklearn import cross_validation

import numpy as np

import matplotlib.pyplot as plt
from frozendict import frozendict
from sqlalchemy.exc import OperationalError

from schlichtanders.mymatplotlib import Centre
from matplotlib.colors import LogNorm
import cPickle as pickle
import os, platform, sys
from pprint import pprint
from sqlalchemy import Column, Integer, Unicode, UnicodeText, String, PickleType, Float, Boolean
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker, Session, create_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.automap import automap_base
import operator as op
from collections import defaultdict, Counter, OrderedDict
import csv
import heapq
from copy import copy
import warnings
import experiment_toy_models
import experiment_models
import experiment_util
from schlichtanders.mycontextmanagers import ignored
from schlichtanders.myobjects import Namespace
from schlichtanders.mygenerators import product

import experiment_util
import numpy as np
import theano


__file__ = os.path.realpath('__file__')
if platform.system() == "Windows":
    from schlichtanders.myos import replace_unc
    __file__ = replace_unc(__file__)
__path__ = os.path.dirname(__file__)
__parent__ = os.path.dirname(__path__)

sys.path.append(__parent__)
import experiment1.experiment_util as ex1util


warnings.filterwarnings("ignore", category=DeprecationWarning)
inf = float('inf')
__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'



# FIND MODES
# ==========



def zero_around(hist):
    return np.r_[0, hist, 0]

def bottom_top(hist):
    histd = np.diff(hist)
    histds = np.sign(histd)
    # treat zeros as the one before, or 1 if it is the very start
    # as one, a subsequent decrease is the first bottom
    last_non_zero = 1
    for i, h in enumerate(histds):
        if h==0:
            histds[i] = last_non_zero
        else:
            last_non_zero = histds[i]

    histdsd = np.diff(histds)
    histbb = histdsd > 0
    histbb = np.r_[False, histbb, False]
    histbt = histdsd < 0
    histbt = np.r_[False, histbt, False]
    # bottom = hist[histbb].tolist()
    # top = hist[histbt].tolist()
    # return bottom, top
    return hist[histbb], hist[histbt]


def start_top(hist, i=0):
    if hist[i] < hist[i+1]:
        return False
    elif hist[i] > hist[i+1]:
        return True
    elif hist[i] == hist[i+1]:
        return start_top(i+1)

def end_top(hist, i=-1):
    if hist[i-1] > hist[i]:
        return False
    elif hist[i-1] < hist[i]:
        return True
    elif hist[i-1] == hist[i]:
        return end_top(i-1)


def first_step(hist, bottom, top):
    true_top = []
    top_slopes = []
    bottom_iter = iter(bottom)
    top_iter = iter(top)
    if start_top(hist):
        true_top.insert(0, hist[0])
        b1 = next(bottom_iter)
    else:
        b1 = hist[0]

    while True:
        try:
            t = next(top_iter)
        except StopIteration:
            # this is always the case, as we finish the loop when finding a bottom end, hence here there must be top end
            true_top.append(hist[-1])
            break
        try:
            b2 = next(bottom_iter)
        except StopIteration:
            # this is always the case, as if there would be a botton in between
            # (an end top still following), then this would have been listed in bottom_iter
            # hence the end is a bottom
            b2 = hist[-1]
            top_slopes.append((t, b1, b2))
            break
        #else continue counting the tops
        top_slopes.append((t, b1, b2))
        b1 = b2
    return true_top, top_slopes


def second_step(top_slopes, threshold_d=40):
    true_top=[]
    b1, t, b2 = None, None, None
    for _t, _b1, _b2 in top_slopes:
        if t is None:
            b1, t, b2 = _b1, _t, _b2
        else:
            if _t > t:
                t = _t
            b2 = _b2  # in any case, as we must take the rightest end as the end

        if t-b1 > threshold_d and t-b2 > threshold_d:
            true_top.append(t)
            b1, t, b2 = None, None, None
    return true_top


def get_modes(hist, threshold_d=40, start_end_bottom=True):
    hist = zero_around(hist)
    if start_end_bottom:
        threshold_d = min(threshold_d, int(0.9*hist.max()))
    bottom, top = bottom_top(hist)
    true_top, top_slopes = first_step(hist, bottom, top)
    true_top += second_step(top_slopes, threshold_d=threshold_d)
    return true_top



# EVALUATION
# =============

# helpers
# -------
def defaultdictdict():
    return defaultdict(dict)

def defaultdictdictdict():
    return defaultdictdict()

def defaultdictlist():
    return defaultdict(list)

def defaultdictdictlist():
    return defaultdict(defaultdictlist)

def tuple4list():
    return [], [], [], []

def defaultdicttuplelist():
    return defaultdict(tuple4list)

def defaultdictdicttuplelist():
    return defaultdict(defaultdicttuplelist)

def defaultdictlist():
    return defaultdict(list)

def gen_subfiles(*directories, **kwargs):
    key = kwargs.pop("key", lambda fn, path: True)
    for d in directories:
        for f in os.walk(d):
            for fn in f[2]:
                if fn.endswith(".db"):
                    path = os.path.join(__path__, f[0], fn)
                    if key(fn, path):
                        yield path


# get best hyperparameters
# ------------------------

def fmap_results(f, results, inplace=False):
    new = results if inplace else {} #defaultdict(defaultdictdictdict)
    for test in results:
        if not inplace:
            new[test] = {}
        for name in results[test]:
            if not inplace:
                new[test][name] = {}
            for percent in results[test][name]:
                if not inplace:
                    new[test][name][percent] = {}
                for nn in results[test][name][percent]:
                    new[test][name][percent][nn] = f(results[test][name][percent][nn])
    return new

def reduce_results(f, results, acc=0):
    for test in results:
        for name in results[test]:
            for percent in results[test][name]:
                for nn in results[test][name][percent]:
                    acc = f(acc, results[test][name][percent][nn])
    return acc


def to_pandas_dict(best_hypers, keys=("datasetname", "modelname", "percent", "n_normflows", "best_val_loss", "best_val_error"),
                   pandas_dict=None):
    if pandas_dict is None:
        pandas_dict = {}
        for k in keys:
            pandas_dict[k] = []

    for hyper in best_hypers:
        for k in keys:
            try:
                pandas_dict[k].append(getattr(hyper, k))
            except KeyError:
                try:
                    pandas_dict[k].append(getattr(best_hypers[hyper], k))
                except (TypeError, KeyError):
                    pandas_dict[k].append(None)  # we must ensure same length everywhere
    return pandas_dict

# -------------------------------
def version_fix(best, prefix):
    if not hasattr(best, "modelname"):
        best.modelname = prefix
    if not hasattr(best, "best_parameters"):
        setattr(best, "best_parameters", getattr(best, prefix + "_best_parameters"))
    if not hasattr(best, "init_parameters"):
        setattr(best, "init_parameters", getattr(best, prefix + "_init_params"))
    if not hasattr(best, "best_val_loss"):
        setattr(best, "best_val_loss", getattr(best, prefix + "_best_val_loss"))
    if not hasattr(best, "val_loss"):
        setattr(best, "val_loss", getattr(best, prefix + "_val_loss"))
    if not hasattr(best, "best_val_error"):
        setattr(best, "best_val_error", getattr(best, prefix + "_val_error_rate"))
    try:
        if not hasattr(best, "best_test_loss"):
            setattr(best, "best_test_loss", getattr(best, prefix + "_best_test_loss"))
        if not hasattr(best, "best_test_error"):
            setattr(best, "best_test_error", getattr(best, prefix + "_test_error_rate"))
    except AttributeError:
        pass
    return best


def get_key_hyper(attr):
    def key_hyper(h):
        if h.best_parameters is None:
            return inf
        val = getattr(h, attr)
        if val == -inf:
            return inf
        return val
    return key_hyper


def get_single_best_hyper(folders, modelname, Hypers=None, attr="best_val_loss", key_files=lambda fn, path:True, filter_hyper=lambda h: True):
    all_data = []
    for f in gen_subfiles(*folders, key=key_files):
        engine = create_engine('sqlite:///' + f)  # echo=True
        if Hypers is None:
            Base = automap_base()
            Base.prepare(engine, reflect=True)
            Hyper = Base.classes.hyper

            session = Session(engine)
            all_data += [version_fix(copy(h), modelname) for h in session.query(Hyper) if filter_hyper(h)]
        else:
            for _Hyper in Hypers:
                try:
                    _Hyper.metadata.create_all(engine)
                    Hyper = _Hyper
                    session = Session(engine)
                    all_data += [version_fix(copy(h), modelname) for h in session.query(Hyper) if filter_hyper(h)]
                    break
                except OperationalError:
                    continue

    best = heapq.nsmallest(1, all_data, key=get_key_hyper(attr))[0]
    return best


def get_best_hyper(folders, Hypers=None, modelnames=('baselinedet', 'baseline', 'planarflow',
                                                   'planarflowdet', 'radialflow', 'radialflowdet'),
                   percentages=(0.25, 0.5, 1.0), n_normflows=(5,10,20,30), test_attrs=("best_val_loss", "val_error_rate"), key_files=lambda fn, path:True):
    best_hypers = []
    all_hypers = []
    for f in gen_subfiles(*folders, key=key_files):
        engine = create_engine('sqlite:///' + f)  # echo=True
        if Hypers is None:
            Base = automap_base()
            Base.prepare(engine, reflect=True)
            Hyper = Base.classes.hyper

            session = Session(engine)
            all_hypers += session.query(Hyper).all()
        else:
            for Hyper in Hypers:
                try:
                    Hyper.metadata.create_all(engine)
                    session = Session(engine)
                    all_hypers += session.query(Hyper).all()
                    break
                except OperationalError:
                    continue

    if percentages is None:
        percentages = [None]
    if n_normflows is None:
        n_normflows = [None]

    for attr, prefix, percent, nn in product(test_attrs, modelnames, percentages, n_normflows):
        # find best fit
        sub_all_data = []
        for h in all_hypers:
            b_nn = nn is None or h.n_normflows == nn
            b_percent = percent is None or h.percent == percent
            if b_nn and b_percent:
                # reformot old version to new one:
                sub_all_data.append(version_fix(copy(h), prefix))

        if sub_all_data:
            best = heapq.nsmallest(1, sub_all_data, key=get_key_hyper(attr))[0]  # only the very best is wanted to keep it simple and clean
            common_format = Namespace({k: v for k, v in best.__dict__.iteritems() if k[:3] not in ["bas", "mix", "pla", "rad"]})  # namespace ensures standard instance access
            best_hypers.append(common_format)
        # else nothing is appended
    return best_hypers  # check for duplicates (might be the case in old hyper representation)


def get_best_hyper_autofix(datasetname, folders_parameters, test_attrs=['best_val_loss', 'best_val_error'],
                           modelnames=("planarflow", "planarflowdet", "radialflow", "radialflowdet"),
                           percentages=None, n_normflows=None):
    # # Collect models and find best ones
    # best_hyper = eva.get_best_hyper(["toy_windows", "toy_linux"], Hyper, model_prefixes, test_suffix=["best_val_loss"])
    modelnames_to_optimization_type = {  # sorted by optimization type
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

    if "toy" in datasetname:
        Hypers = ex1util.get_toy_hyper(), ex1util.get_toy_semiold_hyper(), ex1util.get_toy_old_hyper(), experiment_util.Hyper

        dim = 1 if "1d" in datasetname else 2
        example_input = np.array([0.0] * dim, dtype=theano.config.floatX)
        example_output = [0.0]
        output_transfer = None

    else:
        Hypers = ex1util.get_hyper(), ex1util.get_semiold_hyper(), ex1util.get_old_hyper(), experiment_util.Hyper

        data, error_func = experiment_util.load_and_preprocess_data(datasetname)
        X, Z, VX, VZ, TX, TZ = data
        example_input = X[0]
        example_output = Z[0]
        output_transfer = "softmax" if datasetname == "mnist" else "identity"

    # compute everything:
    if modelnames is None:
        modelnames = modelnames_to_optimization_type.keys()
    best_hypers = get_best_hyper(folders_parameters, Hypers,
                                 modelnames=modelnames,
                                 percentages=percentages,
                                 n_normflows=n_normflows,
                                 test_attrs=test_attrs, key_files=lambda fn, p: datasetname in fn)

    # further unify hyper representation
    for h in best_hypers:
        h.example_input = example_input
        h.example_output = example_output
        h.output_transfer = output_transfer
        h.optimization_type = modelnames_to_optimization_type[h.modelname]
        h.datasetname = datasetname

    return best_hypers


def compute_test_results(best_hypers, data_gen, filepath, n_trials=20, include_best_hyper=True):
    """

    Parameters
    ----------
    best_hypers
    data_gen
    extra_model_args : str or tuple
        either "toy",
        or (example_input, example_output, output_transfer)
        where output_transfer in ["linear", "softmax"]
    n_trials

    Returns
    -------

    """
    if include_best_hyper:
        n_trials -= 1  # best_hyper is regarded as first trial
    try:
        best_tests = load_dict(filepath)
    except IOError:
        best_tests = OrderedDict()

    for hyper in best_hypers:
        if include_best_hyper:
            test_results = [(getattr(hyper, "test_error"),
                             getattr(hyper, "best_test_loss"),
                             getattr(hyper, "best_epoch"),
                             getattr(hyper, "best_parameters"))]
        else:
            test_results = []
        for _ in xrange(n_trials):
            extra_dict = {k: v for k, v in hyper.__dict__.iteritems() if k[:4] not in ["best"] or k in ["val_loss", "train_loss"]}
            with experiment_util.log_exceptions(filepath + "errors.txt", hyper, extra_dict):
                data, error_func = data_gen(hyper)
                model_module = experiment_toy_models if "toy" in hyper.datasetname else experiment_models
                model, approx_posterior = getattr(model_module, hyper.modelname)(hyper)
                test_results.append(experiment_util.optimize(
                    data, hyper, model, error_func
                ))
        test_error_rate, best_test_loss, best_epoch, best_params = zip(*test_results)
        test_results_dict = {
            'test_error_rate': np.array(test_error_rate),
            'best_test_loss': np.array(best_test_loss),
            'best_epoch': np.array(best_epoch),
            'best_parameters': best_params,  # leave this as a list
        }
        best_tests[hyper] = test_results_dict

        with open(filepath, "wb") as f:
            pickle.dump(best_tests, f, -1)
    return best_tests


def sample_best_hyper(best_tests, filepath, n_samples=1000):
    try:
        best_hyper_samples = load_dict(filepath)
    except IOError:
        best_hyper_samples = OrderedDict()

    for hyper in best_tests:
        parameters = best_tests[hyper]["best_parameters"]  # already include the results of hyper
        best_hyper_samples[hyper] = []
        for params in parameters:
            if params is None:  # best_hyper[test][name][nn][0][ihyper] == inf:
                continue  # both test should in principal find the same error
                # this means, the best solution with parameters was still infinite, we have to skip it
            model_module = experiment_toy_models if "toy" in hyper.datasetname else experiment_models
            model, approx_posterior = getattr(model_module, hyper.modelname)(hyper)
            flat = experiment_util.standard_flat(model)
            sampler = theano.function([], approx_posterior['outputs'],
                                      givens={flat: params})  # reduces amount of runtime
            best_hyper_samples[hyper].append(np.array([sampler() for _ in xrange(n_samples)]))
        with open(filepath, "wb") as f:
            pickle.dump(best_hyper_samples, f, -1)


# Test error
# ==========
def load_dict(filepath):
    with open(filepath, "rb") as f:
        results_dict = pickle.load(f)
    return results_dict


def load_test_results(datasetname):
    save_fn = "%s.pkl" % datasetname
    save_path = os.path.join(__path__, "eval_test", save_fn)
    with open(save_path, "rb") as f:
        results_dict = pickle.load(f)
    return results_dict

# Modes
# -----

def get_best_modes(leaf, threshold_d=40):
    new_leaf = []
    for samples in leaf:
        # each column of samples stands for a parameter
        for c in xrange(samples.shape[1]):
            hist = np.histogram(samples[:,c], bins="auto")[0]
            new_leaf.append(get_modes(hist, threshold_d=threshold_d))  # just append all
    return new_leaf


def get_nr_modes_(leaf):
    return Counter(map(len, leaf))

def get_nr_modes(leaf, threshold_d=40):
    return get_nr_modes_(get_best_modes(leaf, threshold_d=threshold_d))

# Correlations
# ------------


def get_best_correlations(leaf):
    new_leaf = []
    for samples in leaf:
        corr = np.corrcoef(samples, rowvar=0)
        new_leaf.append(corr)
    return new_leaf


# KL divergence
# -------------
'''
def kl_hist(hist1, hist2):
    """ density=True must have been used """
    maybe_nans = hist1*np.log(hist1/hist2)
    maybe_nans[np.isnan(maybe_nans)] = 0
    maybe_nans[np.isinf(maybe_nans)] = 0
    return maybe_nans.sum()

def compute_kl(best_hyper_samples):
    test_attrs = sorted(best_hyper_samples.keys())
    model_prefixes = sorted(best_hyper_samples[test_attrs[0]].keys())
    ln = len(model_prefixes)
    percentages = sorted(best_hyper_samples[test_attrs[0]][model_prefixes[0]].keys())
    n_normflows = sorted(best_hyper_samples[test_attrs[0]][model_prefixes[0]][percentages[0]].keys())  # assumed to be the same overall
    n_best = len(best_hyper_samples[test_attrs[0]][model_prefixes[0]][percentages[0]][n_normflows[0]])  # assumed to be the same overall
    dimensionality = best_hyper_samples[test_attrs[0]][model_prefixes[0]][percentages[0]][n_normflows[0]][1].shape[1] # nr columns is dimensionality
    best_kl = {}
    for test in best_hyper_samples:
        best_kl[test] = {}
        for percent in percentages:
            best_kl[test][percent] = {}
            for nn in n_normflows:
                best_kl[test][percent][nn] = []
                for i in xrange(n_best):
                    kl_matrix = np.zeros((ln, ln))
                    for r, name1 in enumerate(model_prefixes):
                        for c, name2 in enumerate(model_prefixes):
                            samples1 = best_hyper_samples[test][name1][nn][i]
                            samples2 = best_hyper_samples[test][name2][nn][i]
                            if dimensionality == 1:
                                hist1, edges = np.histogram(samples1, bins="auto", density=True)
                                hist2, _ = np.histogram(samples2, bins=edges, density=True)
                                kl_matrix[r,c] = kl_hist(hist1, hist2)
                            elif dimensionality == 2:
                                _, edges_x = np.histogram(samples1[:,0], bins="auto", density=True)
                                _, edges_y = np.histogram(samples1[:,1], bins="auto", density=True)
                                hist1, _, _ = np.histogram2d(samples1[:,0], samples1[:,1], bins=(edges_x, edges_y))
                                hist2, _, _ = np.histogram2d(samples2[:,0], samples1[:,1], bins=(edges_x, edges_y))
                                kl_matrix[r, c] = kl_hist(hist1, hist2)
                            else:
                                ValueError("Dimensions > 2 are not supported due to numerical issues")
                    best_kl[test][percent][nn].append(kl_matrix)
    return best_kl
'''