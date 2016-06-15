#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import operator as op
from collections import Sequence
import theano
from theano import gof
from schlichtanders.mydicts import PassThroughDict, DefaultDict
from schlichtanders.myfunctools import fmap

from subgraphs import norm_distance, L2


__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Postmaps
========
Postmaps are meant to be applied just before the model is used somewhere else, e.g. within an optimizer
(Note, until now all postmaps are used for interfacing the optimizers).
As a model is essentially a dictionary, postmaps are mapping a dictionary to a new dictionary. Hence, they are
composable, which is included used in the base implementation of Model.

As the optimizer only needs access to the keys, but won't iterate over it, the ``schlichtanders.mydicts.IdentityDict``
is a well suited type for postmaps to return.

As kwargs of composable functions can be set if the composed function is called, e.g. right before optimization,
further kwargs of the postmaps are very comfortable to set.

"""


"""
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
    IdentityDict over model with standard optimizer keys
    """
    if isinstance(model['outputs'], gof.graph.Variable):
        targets = [model['outputs'].type("deterministic_target")]
        outputs = [model['outputs']]
    else:
        targets = [o.type() for o in model['outputs']]
        outputs = model['outputs']

    return PassThroughDict(model,
        loss_inputs=targets + model['inputs'],
        loss=distance(targets, outputs)
    )


def probabilistic_optimizer_postmap(model):
    """ builds premap for a standard probabilistic model

    Parameters
    ----------
    model : Model
        to be transformed

    Returns
    -------
    IdentityDict over model with standard optimizer keys
    """
    # virtual random variable
    # (we cannot use model['RV'] itself, as the automatic gradients will get confused because model['RV'] is a complex sampler)
    RV = model['outputs'].type("probabilistic_target")  # like targets for deterministic model
    return PassThroughDict(model,
        loss_inputs=[RV] + model['inputs'],
        loss=-model['logP'](RV)
    )


"""
Annealing Postmaps
------------------

Adding a regularizer is a common procedure to improve generalizability of the model (which is usually what we want).
The following define such postmaps.
"""


def regularizer_postmap(model, regularizer_norm=L2, regularizer_scalar=1):
    """ postmap for a standard deterministic model. Simply add this postmap to the model.

    Parameters
    ----------
    model : Model
        kwargs to be adapted by this postmap
    regularizer_norm : function working on list of parameters, returning scalar loss
        shall regularize parameters
    regularizer_scalar : scalar
        weight of loss_regularizer (loss_data is weighted with 1)

    Returns
    -------
    IdentityDict over model
    """
    return PassThroughDict(model,
        loss_data=model['loss'],
        loss_regularizer=regularizer_norm(model['parameters']),
        loss=model['loss'] + regularizer_scalar*regularizer_norm(model['parameters'])
    )


def variational_postmap(model):
    """use this postmap INSTEAD of the standard probabilistic postmap"""
    RV = model['outputs'].type("probabilistic_target")  # like targets for deterministic model
    return PassThroughDict(model,
        loss_inputs=[RV] + model['inputs'],
        loss=-model['logP'](RV),
        loss_data=-model['loglikelihood'](RV),
        loss_regularizer=1/model['n_data'] * model['kl_prior']
    )

"""
Numericalize Postmaps
---------------------
"""


def flat_numericalize_postmap(model, flat_key="flat", mode=None,
                              annealing=False, wrapper=None, wrapper_kwargs={},
                              save_compiled_functions=True, initial_inputs=None, adapt_init_params=lambda ps: ps,
                              profile=False):
    """ postmap to offer an interface for standard numerical optimizer

    'loss' and etc. must be available in the model

    Parameters
    ----------
    model : Model
    annealing : bool
        indicating whether 'loss_data' and 'loss_regularizer' should be used (annealing=True) or 'loss' (default)
    wrapper : function f -> f where f function like used in scipy.optimize.minimize
        wrappers like in schlichtanders.myoptimizers. E.g. batch, online, chunk...
        or a composition of these
    wrapper_kwargs : dict
        extra kwargs for ``wrapper``
    save_compiled_functions : bool
        If false, functions are compiled on every postmap call anew. If true, they are hashed like in a usual DefaultDict
    initial_inputs : list of values matching model['inputs']
        for parameters which are not grounded, but depend on the input (only needed for initialization)
        NON-OPTIONAL!! (because this hidden behaviour might easily lead to weird bugs)
    adapt_init_params : function numpy-vector -> numpy-vector
        for further control of initial parameters

    Returns
    -------
    DefaultDict over model
    """
    assert (not isinstance(model[flat_key], Sequence)), "need single flat vector"
    if initial_inputs is None:
        raise ValueError("Need ``initial_inputs`` to prevent subtle bugs. If really no inputs are needed, please supply"
                         "empty list [] as kwarg.")
    if wrapper is None:
        if annealing:
            def wrapper(fs, **wrapper_kwargs):
                return fmap(op.add, *fs)
        else:
            def wrapper(f, **wrapper_kwargs):
                return f

    parameters = model[flat_key]
    derivatives = {
        "num_loss": lambda loss: model[loss],
        "num_jacobian": lambda loss: theano.grad(model[loss], parameters),
        "num_hessian": lambda loss: theano.gradient.hessian(model[loss], parameters)
    }

    def function(outputs):
        """ compiles function with signature f(params, *loss_inputs) """
        return theano.function([parameters] + model['loss_inputs'], outputs,
                               on_unused_input="warn", allow_input_downcast=True, profile=profile, mode=mode)

    def numericalize(key):
        try:
            if not annealing:
                return function(derivatives[key]("loss"))
            else:
                return function(derivatives[key]("loss_data")), function(derivatives[key]("loss_regularizer"))
        except (KeyError, TypeError, ValueError):
            raise KeyError("requested key %s not computable" % key)
        except AssertionError:
            # TODO got the following AssertionError which seems to be a bug deep in theano/proxifying theano
            # "Scan has returned a list of updates. This should not happen! Report this to theano-users (also include the script that generated the error)"
            # for now we ignore this
            raise KeyError("Internal Theano AssertionError. Hopefully, this will get fixed in the future.")

    def default_getitem(key):
        if key in derivatives:
            return wrapper(numericalize(key), **wrapper_kwargs)
        else:
            return model[key]

    num_parameters = theano.function(model['inputs'], parameters, on_unused_input='ignore')(*initial_inputs)

    dd = DefaultDict(  # DefaultDict will save keys after they are called the first time
        default_getitem=default_getitem,
        default_setitem=lambda key, value: NotImplementedError("You cannot set items on a numericalize postmap."),  # if this would be noexpand always, we could do it savely, but without not
        num_parameters=adapt_init_params(num_parameters)
    )  # TODO add information about keys in derivatives into DefaultDict
    return dd if save_compiled_functions else dd.noexpand()


"""
Concrete Numeric Optimizer Postmaps
-----------------------------------
"""


def scipy_postmap(model):
    """ extracts kwargs for scipy.optimize.minize as far as available and mandatory

    Parameters
    ----------
    model: Model

    Returns
    -------
    dict
    """
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
    """ extracts kwargs for climin.util.optimizer as far as available and mandatory

    Parameters
    ----------
    model: Model

    Returns
    -------
    dict
    """
    kwargs = {
        "f": model["num_loss"],
        "wrt": model["num_parameters"],
    }
    try:
        kwargs["fprime"] = model["num_jacobian"]
    except KeyError:
        pass

    return kwargs