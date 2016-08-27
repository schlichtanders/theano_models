# coding: utf-8
from __future__ import division

import os, platform
import numpy as np
import theano_models as tm
import theano_models.deterministic_models as dm
import theano_models.probabilistic_models as pm
import warnings
from experiment_util import track

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

# capital, as these construct models
Reparam = tm.as_proxmodel('parameters')(tm.prox_reparameterize)
Flat = tm.as_proxmodel("to_be_randomized")(tm.prox_flatten)


def get_prior(total_size, hyper):
    if hyper.adapt_prior:
        return tm.Merge(pm.Gauss(output_shape=(total_size,), init_var=np.exp(-2 * hyper.minus_log_s1)),
                        parameters=None)
    else:
        # nothing is adapted but everything fixed
        return tm.fix_params(pm.Gauss(output_shape=(total_size,), init_var=np.exp(-2 * hyper.minus_log_s1)))

def get_noisy_neuralnetwork(hyper):
    input = tm.as_tensor_variable(hyper.example_input, name="X")
    predictor = dm.Mlp(
        input=input,
        output_size=len(hyper.example_output),
        output_transfer=hyper.output_transfer,
        hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
        hidden_transfers=["rectifier"] * hyper.n_layers
    )
    if hyper.output_transfer == "identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif hyper.output_transfer == "softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")
    return target_distribution, predictor


# BASELINES
# =========

def baselinedet(hyper):
    # this is extremely useful to tell everything the default sizes
    model = tm.Merge(*get_noisy_neuralnetwork(hyper))
    return model, None


def baseline(hyper):
    # this is extremely useful to tell everything the default sizes
    target_distribution, predictor = get_noisy_neuralnetwork(hyper)
    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    total_size = tm.total_size(targets['to_be_randomized'])
    params = pm.DiagGauss(output_size=total_size)
    prior = get_prior(total_size, hyper)
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    return model, params


# PLANARFLOWS
# ===========

def planarflow(hyper):
    # this is extremely useful to tell everything the default sizes
    target_distribution, predictor = get_noisy_neuralnetwork(hyper)
    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    total_size = tm.total_size(targets['to_be_randomized'])
    params_base = pm.DiagGauss(output_size=total_size)
    normflows = [dm.PlanarTransform() for _ in range(hyper.n_normflows)]
    # LocScaleTransform for better working with PlanarTransforms
    params = params_base
    for transform in normflows:
        params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

    prior = get_prior(total_size, hyper)
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    return model, params


def planarflowdet(hyper):
    # this is extremely useful to tell everything the default sizes
    target_distribution, predictor = get_noisy_neuralnetwork(hyper)
    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    target_normflow = tm.Merge(dm.PlanarTransform(), inputs="to_be_randomized") # rename inputs is crucial!!
    for _ in range(hyper.n_normflows - 1):
        target_normflow = tm.Merge(dm.PlanarTransform(target_normflow), target_normflow)
    # target_normflow = tm.Merge(dm.LocScaleTransform(target_normflow, independent_scale=True), target_normflow)

    total_size = tm.total_size(targets['to_be_randomized'])
    targets['to_be_randomized'] = target_normflow
    targets = tm.Merge(targets, target_normflow)

    params = pm.DiagGauss(output_size=total_size)
    prior = get_prior(total_size, hyper)
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    return model, target_normflow



# RADIALFLOWS
# ===========

def radialflow(hyper):
    # this is extremely useful to tell everything the default sizes
    target_distribution, predictor = get_noisy_neuralnetwork(hyper)
    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    total_size = tm.total_size(targets['to_be_randomized'])
    params_base = pm.DiagGauss(output_size=total_size)
    normflows = [dm.RadialTransform() for _ in range(hyper.n_normflows)]
    # LocScaleTransform for better working with PlanarTransforms
    params = params_base
    for transform in normflows:
        params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

    prior = get_prior(total_size, hyper)
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    return model, params


def radialflowdet(hyper):
    # this is extremely useful to tell everything the default sizes
    target_distribution, predictor = get_noisy_neuralnetwork(hyper)
    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    target_normflow = tm.Merge(dm.RadialTransform(), inputs="to_be_randomized")  # rename inputs is crucial!!
    for _ in range(hyper.n_normflows - 1):
        target_normflow = tm.Merge(dm.RadialTransform(target_normflow), target_normflow)
    # target_normflow = tm.Merge(dm.LocScaleTransform(target_normflow, independent_scale=True), target_normflow)

    total_size = tm.total_size(targets['to_be_randomized'])
    targets['to_be_randomized'] = target_normflow
    targets = tm.Merge(targets, target_normflow)

    params = pm.DiagGauss(output_size=total_size)
    prior = get_prior(total_size, hyper)
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    return model, target_normflow


def mixture(hyper):
    # this is extremely useful to tell everything the default sizes
    target_distribution, predictor = get_noisy_neuralnetwork(hyper)
    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    # the number of parameters comparing normflows and mixture of gaussians match perfectly (the only exception is
    # that we spend an additional parameter when modelling n psumto1 with n parameters instead of (n-1) within softmax
    total_size = tm.total_size(targets['to_be_randomized'])
    mixture_comps = [pm.DiagGauss(output_size=total_size) for _ in range(hyper.n_normflows + 1)]  # +1 for base_model
    params = pm.Mixture(*mixture_comps)
    prior = get_prior(total_size, hyper)
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    return model, params