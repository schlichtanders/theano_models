# coding: utf-8
from __future__ import division

import os, platform
import numpy as np
import theano
import theano.tensor as T
import theano_models as tm
import theano_models.deterministic_models as dm
import theano_models.probabilistic_models as pm
import warnings

inf = float("inf")
warnings.filterwarnings("ignore", category=DeprecationWarning)
tm.inputting_references.update(['to_be_randomized'])

__file__ = os.path.realpath(__file__)
if platform.system() == "Windows":
    from schlichtanders.myos import replace_unc
    __file__ = replace_unc(__file__)
__path__ = os.path.dirname(__file__)
__parent__ = os.path.dirname(__path__)


# HELPERS
# =======

def toy_likelihood(dim=1, w_true=0.0):
    w = tm.as_tensor_variable([w_true]*dim)  #T.vector()
    y = w[0] + 0.6 * T.sin(2*np.pi*(w[0]-0.5))
    if dim == 2:
        y += w[1]
    func = tm.Model(inputs=[w], outputs=y, name="sin")
    return tm.Merge(pm.GaussianNoise(y, init_var=0.07), func, ignore_references={'parameters', 'parameters_positive'}) # 0.07 is well suited for 1-dim problem

# capital, as these construct models
Reparam = tm.as_proxmodel('parameters')(tm.prox_reparameterize)
Flat = tm.as_proxmodel("to_be_randomized")(tm.prox_flatten)



# BASELINES
# =========

def baselinedet(hyper):
    model = tm.Merge(toy_likelihood(hyper.dim), inputs="parameters")
    return model, None

def baseline(hyper):
    targets = toy_likelihood(hyper.dim)
    total_size = tm.total_size(targets['inputs'])
    params = pm.DiagGauss(output_size=total_size)
    prior = tm.fix_params(pm.DiagGauss(output_size=total_size))  # gives init_var = 1
    model = tm.variational_bayes(targets, 'inputs', params, priors=prior)
    return model, params


# PLANARFLOWS
# ===========

def planarflow(hyper):
    targets = toy_likelihood(hyper.dim)
    total_size = tm.total_size(targets['inputs'])
    params_base = pm.DiagGauss(output_size=total_size)
    normflows = [dm.PlanarTransform() for _ in range(hyper.n_normflows)]
    # LocScaleTransform for better working with PlanarTransforms
    params = params_base
    for transform in normflows:
        params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP  # TODO merge does not seem to work correctly

    prior = tm.fix_params(pm.DiagGauss(output_size=total_size))
    model = tm.variational_bayes(targets, 'inputs', params, priors=prior)
    return model, params


def planarflowdet(hyper):
    targets = toy_likelihood(hyper.dim)
    target_normflow = tm.Merge(dm.PlanarTransform(), inputs="to_be_randomized") # rename inputs is crucial!!
    for _ in range(hyper.n_normflows - 1):
        target_normflow = tm.Merge(dm.PlanarTransform(target_normflow), target_normflow)
    # target_normflow = tm.Merge(dm.LocScaleTransform(target_normflow, independent_scale=True), target_normflow)

    total_size = tm.total_size(targets['inputs'])
    targets['inputs'] = target_normflow
    targets = tm.Merge(targets, target_normflow)

    params = pm.DiagGauss(output_size=total_size)
    prior = tm.fix_params(pm.DiagGauss(output_size=total_size))
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    return model, target_normflow


# RADIALFLOWS
# ===========

def radialflow(hyper):
    targets = toy_likelihood(hyper.dim)
    total_size = tm.total_size(targets['inputs'])
    params_base = pm.DiagGauss(output_size=total_size)
    normflows = [dm.RadialTransform() for _ in range(hyper.n_normflows)] # *2 as radial flow needs only half of the parameters
    # LocScaleTransform for better working with PlanarTransforms
    params = params_base
    for transform in normflows:
        params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

    prior = tm.fix_params(pm.DiagGauss(output_size=total_size))
    model = tm.variational_bayes(targets, 'inputs', params, priors=prior)
    return model, params


def radialflowdet(hyper):
    targets = toy_likelihood(hyper.dim)

    target_normflow = tm.Merge(dm.RadialTransform(), inputs="to_be_randomized")  # rename inputs is crucial!!
    for _ in range(hyper.n_normflows - 1): # *2 as radial flow needs only half of the parameters
        target_normflow = tm.Merge(dm.RadialTransform(target_normflow), target_normflow)
    # target_normflow = tm.Merge(dm.LocScaleTransform(target_normflow, independent_scale=True), target_normflow)

    total_size = tm.total_size(targets['inputs'])
    targets['inputs'] = target_normflow
    targets = tm.Merge(targets, target_normflow)

    params = pm.DiagGauss(output_size=total_size)
    prior = tm.fix_params(pm.DiagGauss(output_size=total_size))
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    return model, target_normflow


# MIXTURES
# ========

def mixture(hyper):
    targets = toy_likelihood(hyper.dim)
    # the number of parameters comparing normflows and mixture of gaussians match perfectly (the only exception is
    # that we spend an additional parameter when modelling n psumto1 with n parameters instead of (n-1) within softmax
    total_size = tm.total_size(targets['inputs'])
    mixture_comps = [pm.DiagGauss(output_size=total_size) for _ in range(hyper.n_normflows + 1)]  # +1 for base_model
    params = pm.Mixture(*mixture_comps)
    prior = tm.fix_params(pm.DiagGauss(output_size=total_size))
    model = tm.variational_bayes(targets, 'inputs', params, priors=prior)
    return model, params
