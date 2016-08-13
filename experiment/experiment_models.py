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


# BASELINES
# =========

def baselinedet(hyper, example_input, example_output, output_transfer="identity"):
    # this is extremely useful to tell everything the default sizes
    input = tm.as_tensor_variable(example_input, name="X")

    predictor = dm.Mlp(
        input=input,
        output_size=len(example_output),
        output_transfer=output_transfer,
        hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
        hidden_transfers=["rectifier"] * hyper.n_layers
    )
    if output_transfer=="identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif output_transfer=="softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")
    model = tm.Merge(target_distribution, predictor)

    loss = tm.loss_probabilistic(model)  # TODO no regularizer yet ...

    if output_transfer=="identity":
        # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
        all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                            track.squareplus_inv)
        all_params += model['parameters']
    elif output_transfer=="softmax":
        all_params = model['parameters']
    flat = tm.prox_flatten(tm.prox_center(all_params))

    return model, loss, flat


def baselinedetplus(hyper, example_input, example_output, output_transfer="identity"):
    # this is extremely useful to tell everything the default sizes
    input = tm.as_tensor_variable(example_input, name="X")

    predictor = dm.Mlp(
        input=input,
        output_size=len(example_output),
        output_transfer=output_transfer,
        hidden_sizes=[(hyper.units_per_layer+ hyper.units_per_layer*(hyper.n_normflows*2))] * hyper.n_layers,
        hidden_transfers=["rectifier"] * hyper.n_layers
    )
    if output_transfer=="identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif output_transfer=="softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")
    model = tm.Merge(target_distribution, predictor)

    loss = tm.loss_probabilistic(model)  # TODO no regularizer yet ...

    # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
    all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                        track.squareplus_inv)
    all_params += model['parameters']
    flat = tm.prox_flatten(tm.prox_center(all_params))
    return model, loss, flat


def baseline(hyper, example_input, example_output, output_transfer="identity"):
    # this is extremely useful to tell everything the default sizes
    input = tm.as_tensor_variable(example_input, name="X")

    predictor = dm.Mlp(
        input=input,
        output_size=len(example_output),  #Z.shape[1]
        output_transfer=output_transfer,
        hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
        hidden_transfers=["rectifier"] * hyper.n_layers
    )
    if output_transfer=="identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif output_transfer=="softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")
    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    _total_size = tm.total_size(targets['to_be_randomized'])
    params = pm.DiagGauss(output_size=_total_size)
    prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s1)))
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    loss = tm.loss_variational(model)

    # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
    all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                        track.squareplus_inv)
    all_params += model['parameters']
    flat = tm.prox_flatten(tm.prox_center(all_params))
    return model, loss, flat


def baselineplus(hyper, example_input, example_output, output_transfer="identity"):
    # this is extremely useful to tell everything the default sizes
    input = tm.as_tensor_variable(example_input, name="X")

    predictor = dm.Mlp(
        input=input,
        output_size=len(example_output),
        output_transfer=output_transfer,
        hidden_sizes=[hyper.units_per_layer_plus] * hyper.n_layers,
        hidden_transfers=["rectifier"] * hyper.n_layers
    )
    if output_transfer=="identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif output_transfer=="softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")
    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    _total_size = tm.total_size(targets['to_be_randomized'])
    params = pm.DiagGauss(output_size=_total_size)
    prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s1)))
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    loss = tm.loss_variational(model)

    # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
    all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus, track.squareplus_inv)
    all_params += model['parameters']
    flat = tm.prox_flatten(tm.prox_center(all_params))
    return model, loss, flat


# PLANARFLOWS
# ===========

def planarflow(hyper, example_input, example_output, output_transfer="identity"):
    # this is extremely useful to tell everything the default sizes
    input = tm.as_tensor_variable(example_input, name="X")

    predictor = dm.Mlp(
        input=input,
        output_size=len(example_output),
        output_transfer=output_transfer,
        hidden_sizes=[hyper.units_per_layer]*hyper.n_layers,
        hidden_transfers=["rectifier"]*hyper.n_layers
    )
    if output_transfer=="identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif output_transfer=="softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")

    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    _total_size = tm.total_size(targets['to_be_randomized'])
    params_base = pm.DiagGauss(output_size=_total_size)
    normflows = [dm.PlanarTransform() for _ in range(hyper.n_normflows)]
    # LocScaleTransform for better working with PlanarTransforms
    params = params_base
    for transform in normflows:
        params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

    prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s1)))
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    loss = tm.loss_variational(model)

    # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
    all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus, track.squareplus_inv)
    all_params += model['parameters']
    flat = tm.prox_flatten(tm.prox_center(all_params))
    return model, loss, flat


def planarflowdet(hyper, example_input, example_output, output_transfer="identity"):
    # this is extremely useful to tell everything the default sizes
    input = tm.as_tensor_variable(example_input, name="X")

    predictor = dm.Mlp(
        input=input,
        output_size=len(example_output),
        output_transfer=output_transfer,
        hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
        hidden_transfers=["rectifier"] * hyper.n_layers
    )
    if output_transfer=="identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif output_transfer=="softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")

    target_normflow = tm.Merge(dm.PlanarTransform(), inputs="to_be_randomized") # rename inputs is crucial!!
    for _ in range(hyper.n_normflows - 1):
        target_normflow = tm.Merge(dm.PlanarTransform(target_normflow), target_normflow)
    # target_normflow = tm.Merge(dm.LocScaleTransform(target_normflow, independent_scale=True), target_normflow)

    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))
    _total_size = tm.total_size(targets['to_be_randomized'])
    targets['to_be_randomized'] = target_normflow
    targets = tm.Merge(targets, target_normflow)

    params = pm.DiagGauss(output_size=_total_size)
    prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s1)))
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    loss = tm.loss_variational(model)

    # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
    all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus, track.squareplus_inv)
    all_params += model['parameters']
    flat = tm.prox_flatten(tm.prox_center(all_params))
    return model, loss, flat



def planarflowml(hyper, example_input, example_output, output_transfer="identity"):
    # this is extremely useful to tell everything the default sizes
    input = tm.as_tensor_variable(example_input, name="X")

    predictor = dm.Mlp(
        input=input,
        output_size=len(example_output),
        output_transfer=output_transfer,
        hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
        hidden_transfers=["rectifier"] * hyper.n_layers
    )
    if output_transfer=="identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif output_transfer=="softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")

    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    params_base = pm.DiagGauss(output_size=tm.total_size(targets['to_be_randomized']))
    normflows = [dm.PlanarTransform() for _ in range(hyper.n_normflows)]  # no LocScaleTransform
    # LocScaleTransform for better working with PlanarTransforms
    params = params_base
    for transform in normflows:
        params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

    targets['to_be_randomized'] = params
    model = tm.Merge(targets, params)
    loss = tm.loss_probabilistic(model)

    # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
    all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                        track.squareplus_inv)
    all_params += model['parameters']
    flat = tm.prox_flatten(tm.prox_center(all_params))
    return model, loss, flat


# RADIALFLOWS
# ===========

def radialflow(hyper, example_input, example_output, output_transfer="identity"):
    # this is extremely useful to tell everything the default sizes
    input = tm.as_tensor_variable(example_input, name="X")

    predictor = dm.Mlp(
        input=input,
        output_size=len(example_output),
        output_transfer=output_transfer,
        hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
        hidden_transfers=["rectifier"] * hyper.n_layers
    )
    if output_transfer=="identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif output_transfer=="softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")

    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    _total_size = tm.total_size(targets['to_be_randomized'])
    params_base = pm.DiagGauss(output_size=_total_size)
    normflows = [dm.RadialTransform() for _ in range(hyper.n_normflows*2)] # *2 as radial flow needs only half of the parameters
    # LocScaleTransform for better working with PlanarTransforms
    params = params_base
    for transform in normflows:
        params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

    prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s1)))
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    loss = tm.loss_variational(model)

    # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
    all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                        track.squareplus_inv)
    all_params += model['parameters']
    flat = tm.prox_flatten(tm.prox_center(all_params))
    return model, loss, flat


def radialflowdet(hyper, example_input, example_output, output_transfer="identity"):
    # this is extremely useful to tell everything the default sizes
    input = tm.as_tensor_variable(example_input, name="X")

    predictor = dm.Mlp(
        input=input,
        output_size=len(example_output),
        output_transfer=output_transfer,
        hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
        hidden_transfers=["rectifier"] * hyper.n_layers
    )
    if output_transfer=="identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif output_transfer=="softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")

    target_normflow = tm.Merge(dm.RadialTransform(), inputs="to_be_randomized")  # rename inputs is crucial!!
    for _ in range(hyper.n_normflows*2 - 1): # *2 as radial flow needs only half of the parameters
        target_normflow = tm.Merge(dm.RadialTransform(target_normflow), target_normflow)
    # target_normflow = tm.Merge(dm.LocScaleTransform(target_normflow, independent_scale=True), target_normflow)

    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))
    _total_size = tm.total_size(targets['to_be_randomized'])
    targets['to_be_randomized'] = target_normflow
    targets = tm.Merge(targets, target_normflow)

    params = pm.DiagGauss(output_size=_total_size)
    prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s1)))
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    loss = tm.loss_variational(model)

    # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
    all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                        track.squareplus_inv)
    all_params += model['parameters']
    flat = tm.prox_flatten(tm.prox_center(all_params))
    return model, loss, flat


def radialflowml(hyper, example_input, example_output, output_transfer="identity"):
    # this is extremely useful to tell everything the default sizes
    input = tm.as_tensor_variable(example_input, name="X")

    predictor = dm.Mlp(
        input=input,
        output_size=len(example_output),
        output_transfer=output_transfer,
        hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
        hidden_transfers=["rectifier"] * hyper.n_layers
    )
    if output_transfer == "identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif output_transfer == "softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")
    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    params_base = pm.DiagGauss(output_size=tm.total_size(targets['to_be_randomized']))
    normflows = [dm.RadialTransform() for _ in
                 range(hyper.n_normflows * 2)]  # *2 as radial flow needs only half of the parameters
    # LocScaleTransform for better working with PlanarTransforms
    params = params_base
    for transform in normflows:
        params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

    targets['to_be_randomized'] = params
    model = tm.Merge(targets, params)
    loss = tm.loss_probabilistic(model)

    # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
    all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                        track.squareplus_inv)
    all_params += model['parameters']
    flat = tm.prox_flatten(tm.prox_center(all_params))
    return model, loss, flat


# MIXTURES
# ========

def mixture(hyper, example_input, example_output, output_transfer="identity"):
    # this is extremely useful to tell everything the default sizes
    input = tm.as_tensor_variable(example_input, name="X")

    predictor = dm.Mlp(
        input=input,
        output_size=len(example_output),
        output_transfer=output_transfer,
        hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
        hidden_transfers=["rectifier"] * hyper.n_layers
    )
    if output_transfer == "identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif output_transfer == "softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")
    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    # the number of parameters comparing normflows and mixture of gaussians match perfectly (the only exception is
    # that we spend an additional parameter when modelling n psumto1 with n parameters instead of (n-1) within softmax
    _total_size = tm.total_size(targets['to_be_randomized'])
    mixture_comps = [pm.DiagGauss(output_size=_total_size) for _ in range(hyper.n_normflows + 1)]  # +1 for base_model
    params = pm.Mixture(*mixture_comps)
    prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s1)))
    model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
    loss = tm.loss_variational(model)

    # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
    all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus, track.squareplus_inv)
    all_params += tm.prox_reparameterize(model['parameters_psumto1'], tm.softmax, tm.softmax_inv)
    all_params += model['parameters']
    flat = tm.prox_flatten(tm.prox_center(all_params))
    return model, loss, flat


def mixtureml(hyper, example_input, example_output, output_transfer="identity"):
    # this is extremely useful to tell everything the default sizes
    input = tm.as_tensor_variable(example_input, name="X")

    predictor = dm.Mlp(
        input=input,
        output_size=len(example_output),
        output_transfer=output_transfer,
        hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
        hidden_transfers=["rectifier"] * hyper.n_layers
    )
    if output_transfer == "identity":
        target_distribution = pm.DiagGaussianNoise(predictor)
    elif output_transfer == "softmax":
        target_distribution = pm.Categorical(predictor)
    else:
        raise ValueError("can only describe identity or softmax output")
    targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

    _total_size = tm.total_size(targets['to_be_randomized'])
    mixture_comps = [pm.DiagGauss(output_size=_total_size) for _ in
                     range(hyper.n_normflows + 1)]  # +1 for base_model
    params = pm.Mixture(*mixture_comps)

    targets['to_be_randomized'] = params
    model = tm.Merge(targets, params)
    loss = tm.loss_probabilistic(model)

    # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
    all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus, track.squareplus_inv)
    all_params += tm.prox_reparameterize(model['parameters_psumto1'], tm.softmax, tm.softmax_inv)
    all_params += model['parameters']
    flat = tm.prox_flatten(tm.prox_center(all_params))
    return model, loss, flat
