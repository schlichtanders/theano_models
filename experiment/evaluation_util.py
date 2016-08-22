#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os, platform, sys
import warnings
import numpy as np

import matplotlib.pyplot as plt
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
from collections import defaultdict, Counter
import csv
import heapq
from copy import copy
import warnings
import experiment_toy_models
import experiment_models
import experiment_util
from schlichtanders.mycontextmanagers import ignored
from schlichtanders.myobjects import Namespace

import experiment_util
import numpy as np
import theano


__file__ = os.path.realpath('__file__')
if platform.system() == "Windows":
    from schlichtanders.myos import replace_unc
    __file__ = replace_unc(__file__)
__path__ = os.path.dirname(__file__)
__parent__ = os.path.dirname(__path__)


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


def to_pandas_dict(datasetname, best_hyper, pandas_dict=None, last_layer_to_dict=None):
    if pandas_dict is None:
        pandas_dict = defaultdict(list, {
            'datasetname': [],
            'test_measures': [],
            'model': [],
            'n_normflows': [],
        })
    if last_layer_to_dict is None:
        def last_layer_to_dict(last_layer):
            return {"value%i"%i: v for i, v in enumerate(last_layer[0])}  # for best_hyper

    for test in best_hyper:
        for name in best_hyper[test]:
            for nn in best_hyper[test][name]:
                pandas_dict['datasetname'].append(datasetname)
                pandas_dict['test_measures'].append(test)
                pandas_dict['model'].append(name)
                pandas_dict['n_normflows'].append(nn)
                pandas_dict.update()
                for k, v in last_layer_to_dict(best_hyper[test][name][nn]).iteritems():
                    pandas_dict[k].append(v)
#                 for i, v in enumerate(best_hyper[test][name][nn][0]):
#                     vn = "value%i" % i
#                     pandas_dict[vn].append(v)
    return pandas_dict

# -------------------------------

def get_best_hyper(folders, Hyper, model_prefixes, percentages=(0.25, 0.5, 1.0), test_suffix=("best_val_loss", "val_error_rate"), key=lambda fn, path:True):
    all_data = []
    for f in gen_subfiles(*folders, key=key): #"toy_windows", "toy_linux"):
        engine = create_engine('sqlite:///' + f)
        Hyper.metadata.create_all(engine)
        session = Session(engine)
        all_data += session.query(Hyper).all()  # filter if you want

    n_normflows = set(h.n_normflows for h in all_data)
    best_hyper = {}
    for suffix in test_suffix:
        best_hyper[suffix] = {}
        for prefix in model_prefixes:
            best_hyper[suffix][prefix] = {}
            def key(h):
                if getattr(h, prefix + "_best_parameters") is None:
                    return inf
                if getattr(h, attr) == -inf:
                    return inf
                return getattr(h, attr)
            for percent in percentages:
                best_hyper[suffix][prefix][percent] = {}
                for nn in n_normflows:
                    attr = prefix + "_" + suffix
                    # find best fit
                    all_data_nn = [h for h in all_data if h.n_normflows == nn and h.percent == percent]
                    hyper = heapq.nsmallest(1, all_data_nn, key=key)[0]  # only the very best is wanted to keep it simple and clean
                    value = getattr(hyper, attr) #map(op.attrgetter(attr), entries)
                    best_hyper[suffix][prefix][percent][nn] = value, hyper
    return best_hyper


def sample_best_hyper(best_hyper, best_tests, filepath, model_module_id="toy", n_samples=1000):
    model_module = experiment_toy_models if model_module_id == "toy" else experiment_models
    try:
        best_hyper_samples = load_dict(filepath)
    except IOError:
        best_hyper_samples = {}
    for test in best_hyper:
        if test not in best_hyper_samples:
            best_hyper_samples[test] = {}
        for name in best_hyper[test]:
            if name not in best_hyper_samples[test]:
                best_hyper_samples[test][name] = {}
            if "baselinedet" in name or "plus" in name:
                continue  # baselinedet has no distribution, and plus has wrong parameters
            for percent in best_hyper[test][name]:
                if percent not in best_hyper_samples[test][name][percent]:
                    best_hyper_samples[test][name][percent] = {}
                for nn in best_hyper[test][name][percent]:  # n_normflows
                    if nn not in best_hyper_samples[test][name][percent]: # otherwise it was computed already
                        best_hyper_samples[test][name][percent][nn] = []
                        hyper = best_hyper[test][name][percent][nn][1]
                        parameters = best_tests[test][name][percent][nn]["best_parameters"]  # already include the results of hyper
                        for params in parameters:
                            if params is None:  # best_hyper[test][name][nn][0][ihyper] == inf:
                                continue  # both test should in principal find the same error
                                # this means, the best solution with parameters was still infinite, we have to skip it
                            if model_module_id == "toy":
                                model, loss, flat, approx_posterior = getattr(model_module, name)(hyper)
                            else:
                                model, loss, flat, approx_posterior = getattr(model_module, name)(hyper, *model_module_id)
                            sampler = theano.function([], approx_posterior['outputs'], givens={flat: params})  # reduces amount of runtime
                            best_hyper_samples[test][name][percent][nn].append( np.array([sampler() for _ in xrange(n_samples)]) )
                        with open(filepath, "wb") as f:
                            pickle.dump(best_hyper_samples, f, -1)
    return best_hyper_samples

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


# TODO include percent into evaluation
def compute_test_results(best_hyper, data_gen, optimization_type, filepath, extra_model_args=tuple(), same_init_params=True, n_trials=20,
                         include_best_hyper=True):
    """

    Parameters
    ----------
    best_hyper
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
    model_module = experiment_models if extra_model_args else experiment_toy_models  # toy model does not need extra args

    try:
        best_tests = load_dict(filepath)
    except IOError:
        best_tests = {}
    for test in best_hyper:
        if test not in best_tests:
            best_tests[test] = {}
        for name in best_hyper[test]:
            if name not in best_tests[test]:
                best_tests[test][name] = {}
            print(name)
            for percent in best_hyper[test][name]:
                if percent not in best_tests[test][name]:
                    best_tests[test][name][percent] = {}
                for nn in best_hyper[test][name][percent]:  # n_normflows
                    if nn not in best_tests[test][name][percent]:
                        best_tests[test][name][percent][nn] = {}
                        # very_best_hyper
                        hyper = best_hyper[test][name][percent][nn][1] # zero refers to the performance value, one to hypers
                        # for hyper in best_hyper[test][name][nn][1]:
                        if include_best_hyper:
                            test_results = [(getattr(hyper, name + "_test_error_rate"),
                                             getattr(hyper, name + "_best_test_loss"),
                                             getattr(hyper, name + "_best_epoch"),
                                             getattr(hyper, name + "_best_parameters"))]
                        else:
                            test_results = []
                        for _ in xrange(n_trials):
                            extra_dict = {k: v for k,v in hyper.__dict___.iteritems() if k[:4] not in ["base", "plan", "radi", "mixt"]}
                            with experiment_util.log_exceptions(filepath + "errors.txt", "%s,%s,%s"%(name, percent, nn), extra_dict):
                                data, error_func = data_gen(hyper)
                                if same_init_params:
                                    init_params = getattr(hyper, name + "_init_params")
                                else:
                                    init_params = None
                                model, loss, flat, approx_posterior = getattr(model_module, name)(hyper, *extra_model_args)
                                test_results.append(experiment_util.test(
                                    data, hyper, model, loss, flat, error_func, optimization_type[name], init_params
                                ))
                        test_error_rate, best_test_loss, best_epoch, best_params = zip(*test_results)
                        best_tests[test][name][percent][nn]['test_error_rate'] = np.array(test_error_rate)
                        best_tests[test][name][percent][nn]['best_test_loss'] = np.array(best_test_loss)
                        best_tests[test][name][percent][nn]['best_epoch'] = np.array(best_epoch)
                        best_tests[test][name][percent][nn]['best_parameters'] = best_params  # leave this as a list
                        with open(filepath, "wb") as f:
                            pickle.dump(best_tests, f, -1)
    return best_tests

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
