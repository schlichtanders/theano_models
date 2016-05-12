#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import theano
from theano import gof
import numpy as np
from itertools import izip


from schlichtanders.mydicts import IdentityDict, DefaultDict
from schlichtanders.mylists import deepflatten
from schlichtanders.mynumpy import complex_reshape

from util.theano_helpers import norm_distance, L2


__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Postmaps
========
Postmaps are meant to be applied just before the graph is used somewhere else, e.g. within an optimizer
(Note, until now all postmaps are used for interfacing the optimizers).
As a model is essentially a dictionary, postmaps are mapping a dictionary to a new dictionary. Hence, they are
composable, which is included used in the base implementation of Model.

As the optimizer only needs access to the keys, but won't iterate over it, the ``schlichtanders.mydicts.IdentityDict``
is a well suited type for postmaps to return.

As kwargs of composable functions can be set if the composed function is called, e.g. right before optimization,
further kwargs of the postmaps are very comfortable to set.

Optimizer Postmaps
------------------
"""


def deterministic_optimizer_postmap(model, distance=norm_distance()):
    """ builds premap for a standard deterministic model

    Parameters
    ----------
    model : Model
        to be transformed
    distance : metric, function working on two lists of theano expressions
        comparing targets (given as extra input for optimizer) with outputs
        Defaults to standard square loss.

    Returns
    -------
    standard premap for optimizer
    """
    if isinstance(model['outputs'], gof.graph.Variable):
        targets = [model['outputs'].type()]
        outputs = [model['outputs']]
    else:
        targets = [o.type() for o in model['outputs']]
        outputs = model['outputs']

    return IdentityDict(
        lambda key: model[key],
        loss_inputs=targets + model['inputs'],
        loss=distance(targets, outputs)
    )


def probabilistic_optimizer_postmap(model):
    # virtual random variable
    # (we cannot use graph['RV'] itself, as the automatic gradients will get confused because graph['RV'] is a complex sampler)
    RV = model['RV'].type()  # like targets for deterministic model
    return IdentityDict(
        lambda key: model[key],
        loss_inputs=[RV] + model['inputs'],
        loss=-model['logP'](RV)
    )


"""
Annealing Postmaps
------------------

Adding a regularizer is a common procedure to improve generalizability of the model (which is usually what we want).
The following define such postmaps.
"""


def regularize_postmap(model, regularizer_norm=L2):
    """ postmap for a standard deterministic model. Simply add this postmap to the graph.

    Parameters
    ----------
    model : Model
        kwargs to be adapted by this postmap
    regularizer_norm : function working on list of parameters, returning scalar loss
        shall regularize parameters

    Returns
    -------
    IdentityDict
    """
    return IdentityDict(
        lambda key: model[key],
        loss_data=model['loss'],
        loss_regularizer=regularizer_norm(model['parameters']),
        loss=model['loss'] + regularizer_norm(model['parameters'])
    )


def variational_postmap(model):
    """use this postmap INSTEAD of the standard probabilistic postmap"""
    RV = model['RV'].type()  # like targets for deterministic model
    return IdentityDict(
        lambda key: model[key],
        loss_inputs=[RV] + model['inputs'],
        loss=-model['logP'](RV),
        loss_data=-model['loglikelihood'](RV),
        loss_regularizer=1/model['n_data'] * model['kl_prior']
    )



"""
numericalization postmaps
-------------------------
"""


def numerical_parameters(model):
    """ standard remap for accessing shared theano parameters

    if graph['parameters'] refers to singleton, then its numerical value is used directly, otherwise the numerical
    values of all parameters are flattened out and concatinated to give a numerical representation alltogether
    """
    num_parameters = []
    # singleton case:
    if len(model['parameters']) == 1:
        # borrow=True is for 1) the case that the optimizer works inplace, 2) its faster
        return model['parameters'][0].get_value(borrow=True)  # return it directly, without packing it into a list
    # else, flatten parameters out (as we cannot guarantee matching shapes):
    else:
        for p in model['parameters']:
            v = p.get_value(borrow=True)  # p.get_value(borrow=True) # TODO what does borrow?
            num_parameters += deepflatten(v)
    return np.array(num_parameters)  # default to numpy type, as this supports numeric operators like indented


def numericalize(model, loss_reference_name, d_order=0):
    """ numericalizes ``graph[loss_reference_name]`` or the respective derivative of order ``d_order``

    It works analogously to ``numerical_parameters`` in that it handles singleton cases or otherwise reshapes flattened
    numerical parameters.

    Parameters
    ----------
    model : Graph
        source from which to wrap
    loss_reference_name: str
        graph[reference_name] will be wrapped
    d_order : int
        order of derivative (0 stands for no derivative) which shall be computed
    """
    # handle singleton parameters:
    parameters = model['parameters'][0] if len(model['parameters']) == 1 else model['parameters']
    if d_order == 0:
        outputs = model[loss_reference_name]
    elif d_order == 1:
        outputs = theano.grad(model[loss_reference_name], parameters)
    elif d_order == 2:
        outputs = theano.gradient.hessian(model[loss_reference_name], parameters)
    else:
        raise ValueError("Derivative of order %s is not yet implemented" % d_order)

    f_theano = theano.function(model['loss_inputs'], outputs)
    shapes = [p.get_value(borrow=True).shape for p in model['parameters']]  # borrow=True as it is faster

    def f(xs, *args, **kwargs):
        if len(model['parameters']) == 1:
            model['parameters'][0].set_value(xs, borrow=True)  # xs is not packed within list
            return f_theano(*args, **kwargs)  # where initialized correctly

        # else, reshape flattened parameters
        else:
            xs = list(complex_reshape(xs, shapes))
            for x, p in izip(xs, model['parameters']):
                p.set_value(x, borrow=True)

            if d_order == 0:
                return f_theano(*args, **kwargs)
            elif d_order == 1:
                # CAUTION: must return array type, as this output type should be averagable and else...
                return np.array(deepflatten(f_theano(*args, **kwargs)))
            else:
                raise ValueError("flat reparameterization of order %s is not yet implemented" % d_order)
                # other orders not yet implemented (will get a bit messy on the hessian,
                # but there for sure is a general solution to this problem)
    return f


def numericalize_postmap(model, annealing=False, wrapper=None, wrapper_kwargs={}):
    """ final postmap to offer an interface for standard numerical optimizer

    'loss' and etc. must be available in the graph"""
    if wrapper is None:
        if not annealing:
            def wrapper(f, **wrapper_kwargs): return f
        else:
            def wrapper(fs, **wrapper_kwargs): return sum(fs)

    d_order = {
        "num_loss": 0,
        "num_jacobian": 1,
        "num_hessian": 2,
    }

    def lazy_numericalize(key):
        if not annealing:
            return numericalize(model, "loss", d_order=d_order[key])
        else:
            return (
                numericalize(model, "loss_data", d_order=d_order[key]),
                numericalize(model, "loss_regularizer", d_order=d_order[key])
            )

    return DefaultDict(  # DefaultDict will save keys after they are called the first time
        lambda key: wrapper(lazy_numericalize(key), **wrapper_kwargs),
        num_parameters=numerical_parameters(model)
    )


"""
concrete numeric optimizer
--------------------------
"""


def scipy_postmap(model):
    kwargs = {
        "fun": model["num_loss"],
        "x0": model["num_parameters"],
    }
    try:
        kwargs["jac"] = model["num_jacobian"]
    except KeyError:
        pass

    try:
        kwargs["hessp"] = model["num_hessianp"]
    except KeyError:
        try:
            kwargs["hess"] = model["num_hessian"]
        except KeyError:
            pass
    return kwargs


def climin_postmap(model):
    kwargs = {
        "f": model["num_loss"],
        "wrt": model["num_parameters"],
    }
    try:
        kwargs["fprime"] = model["num_jacobian"]
    except KeyError:
        pass

    return kwargs