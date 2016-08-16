#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os, platform, sys
import warnings
import numpy as np

import matplotlib.pyplot as plt
from schlichtanders.mymatplotlib import Centre
from matplotlib.colors import LogNorm

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

def defaultdictdictlist():
    def defaultdictlist():
        return defaultdict(list)
    return defaultdict(defaultdictlist)

def defaultdictdicttuplelist():
    def defaultdicttuplelist():
        return defaultdict(lambda: ([], [], [], []))
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

def get_best_hyper(folders, Hyper, model_prefixes, n_best=10, test_suffix=("best_val_loss", "val_error_rate"), key=lambda fn, path:True):
    all_data = []
    for f in gen_subfiles(*folders, key=key): #"toy_windows", "toy_linux"):
        engine = create_engine('sqlite:///' + f)
        Hyper.metadata.create_all(engine)
        session = Session(engine)
        all_data += session.query(Hyper).all()  # filter if you want

    n_normflows = set(h.n_normflows for h in all_data)
    best_hyper = defaultdict(defaultdictdict)
    for prefix in model_prefixes:
        def key(h):
            if getattr(h, prefix + "_best_parameters") is None:
                return inf
            return getattr(h, attr)

        for nn in n_normflows:
            for suffix in test_suffix:
                attr = prefix + "_" + suffix
                # find best fit
                all_data_nn = [h for h in all_data if h.n_normflows == nn]
                entries = heapq.nsmallest(n_best, all_data_nn, key=key)
                values = map(op.attrgetter(attr), entries)
                best_hyper[suffix][prefix][nn] = values, entries
    return best_hyper


def sample_best_hyper(best_hyper, model_module_id="toy", n_samples=1000):
    model_module = experiment_toy_models if model_module_id == "toy" else experiment_models
    best_hyper_samples = defaultdict(defaultdictdictlist)
    for test in best_hyper:
        for name in best_hyper[test]:
            if "baselinedet" in name or "plus" in name:
                continue  # baselinedet has no distribution, and plus has wrong parameters
            for nn in best_hyper[test][name]:  # n_normflows
                for ihyper, hyper in enumerate(best_hyper[test][name][nn][1]):  # hypers
                    params = getattr(hyper, name + "_best_parameters")
                    if params is None: # best_hyper[test][name][nn][0][ihyper] == inf:
                        continue # both test should in principal find the same error
                        # this means, the best solution with parameters was still infinite, we have to skip it
                    if model_module_id == "toy":
                        model, loss, flat, approx_posterior = getattr(model_module, name)(hyper)
                    else:
                        model, loss, flat, approx_posterior = getattr(model_module, name)(hyper, *model_module_id)
                    sampler = theano.function([], approx_posterior['outputs'], givens={flat:params})  # reduces amount of runtime
                    best_hyper_samples[test][name][nn].append( np.array([sampler() for _ in xrange(n_samples)]) )
    return best_hyper_samples

# Test error
# ==========

def compute_test_results(best_hyper, data_gen, model_module_id="toy", n_trials=20):
    """

    Parameters
    ----------
    best_hyper
    data_gen
    model_module_id : str or tuple
        either "toy",
        or (example_input, example_output, output_transfer)
        where output_transfer in ["linear", "softmax"]
    n_trials

    Returns
    -------

    """
    model_module = experiment_toy_models if model_module_id == "toy" else experiment_models
    best_tests = defaultdict(defaultdictdicttuplelist)
    for test in best_hyper:
        for name in best_hyper[test]:
            if name == "baselinedet":
                optimization_type = "ml"
            else:
                optimization_type = "annealing"
            for nn in best_hyper[test][name]:  # n_normflows
                for hyper in best_hyper[test][name][nn][1]:  # one because 0 refers to the performance value
                    test_results = []
                    for _ in xrange(n_trials):
                        data, error_func = data_gen(hyper)
                        init_params = getattr(hyper, name + "_init_params")
                        if model_module_id == "toy":
                            model, loss, flat, approx_posterior = getattr(model_module, name)(hyper)
                        else:
                            model, loss, flat, approx_posterior = getattr(model_module, name)(hyper, *model_module_id)
                        test_results.append(experiment_util.test(
                            data, hyper, model, loss, flat, error_func, optimization_type, init_params
                        ))
                    test_error_rate, best_test_loss, best_epoch, best_params = zip(*test_results)
                    best_tests[test][name][nn][0].append(np.array(test_error_rate))
                    best_tests[test][name][nn][1].append(np.array(best_test_loss))
                    best_tests[test][name][nn][2].append(np.array(best_epoch))
                    best_tests[test][name][nn][3].append(np.array(best_params))
    return best_tests

# Modes
# -----

def get_best_modes(best_hyper_samples):
    best_modes = defaultdict(defaultdictdictlist)
    for test in best_hyper_samples:
        for name in best_hyper_samples[test]:  # model_prefixes
            for nn in best_hyper_samples[test][name]:  # n_normflows
                for samples in best_hyper_samples[test][name][nn]:
                    # each column of samples stands for a parameter
                    for c in xrange(samples.shape[1]):
                        hist = np.histogram(samples[:,c], bins="auto")[0]
                        best_modes[test][name][nn].append(get_modes(hist))  # just append all
    return best_modes


def get_nr_modes_(best_modes):
    best_nr_modes = defaultdict(defaultdictdict)
    for test in best_modes:
        for name in best_modes[test]:
            for nn in best_modes[test][name]:  #n_normflows
                best_nr_modes[test][name][nn] = Counter(map(len, best_modes[test][name][nn]))
    return best_nr_modes

def get_nr_modes(best_hyper_samples):
    best_modes = get_best_modes(best_hyper_samples)
    return get_nr_modes_(best_modes)

# Correlations
# ------------


def get_best_correlations(best_hyper_samples):
    best_correlations = defaultdict(defaultdictdictlist)
    for test in best_hyper_samples:
        for name in best_hyper_samples[test]:  # model_prefixes
            for nn in best_hyper_samples[test][name]:  # n_normflows
                for samples in best_hyper_samples[test][name][nn]:
                    corr = np.corrcoef(samples, rowvar=0)
                    best_correlations[test][name][nn].append(corr)
    return best_correlations


# KL divergence
# -------------

def kl_hist(hist1, hist2):
    """ density=True must have been used """
    maybe_nans = hist1*np.log(hist1/hist2)
    maybe_nans[np.isnan(maybe_nans)] = 0
    maybe_nans[np.isinf(maybe_nans)] = 0
    return maybe_nans.sum()

def compute_kl(best_hyper_samples):
    ex_test_dict = next(best_hyper_samples.itervalues())
    ex_name_dict = next(ex_test_dict.itervalues())
    ex_normflow_list = next(ex_name_dict.itervalues())
    dimensionality = ex_normflow_list[1].shape[1]  # nr columns is dimensionality
    best_kl = defaultdict(defaultdictlist)
    for test in best_hyper_samples:
        model_prefixes = best_hyper_samples[test].keys()
        ln = len(model_prefixes)
        n_normflows = best_hyper_samples[test][model_prefixes[0]].keys()  # assumed to be the same overall
        n_best = len(best_hyper_samples[test][model_prefixes[0]][n_normflows[0]])  # assumed to be the same overall
        for nn in n_normflows:
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
                best_kl[test][nn].append(kl_matrix)
    return best_kl
