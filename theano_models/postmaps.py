#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import operator as op
from collections import Sequence
from itertools import izip, repeat

import numpy as np
import theano
import theano.tensor as T
from schlichtanders.mylists import as_list
from theano import gof
from schlichtanders.mydicts import PassThroughDict, DefaultDict, update
from schlichtanders.myfunctools import fmap

from subgraphs import norm_distance, L2
from theano.gof.fg import MissingInputError
from theano_models.util.theano_helpers import independent_subgraphs
from util import clone_renew_rng


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

from theano.tensor import dvector
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
theano wrapper postmaps
-----------------------
"""


# # Deprecated. Nice idea, however does not work theoretically as theano.clone within theano.scan does not work
# def reduce_postmap(model, reduce_op=T.add):
#     # build new loss_inputs with extra dimension (will be regarded as first dimension)
#     loss_inputs = [T.TensorType(i.dtype, i.broadcastable + (False,))(i.name + ("" if i.name is None else "R"))
#                    for i in model['loss_inputs']]
#
#     def _reduce(key):
#         expr = model[key]
#         if isinstance(expr, Sequence):
#             raise KeyError("Can only reduce references to theano variables, no lists.")
#
#         def fn(*args):
#             # acc, expr = args[-2:]
#             acc = args[-1]
#             rows_loss_inputs = args[:-1]
#             expr_cp = clone_renew_rng(expr, replace=dict(izip(model['loss_inputs'], rows_loss_inputs)))
#             return reduce_op(acc, expr_cp)
#
#         # output_info = np.array(0, expr.dtype)
#         output_info = np.zeros((1,)*expr.ndim, expr.dtype)
#         return theano.reduce(fn, loss_inputs, output_info)[0]  # we don't need updates [1]
#
#     if 'loss_data' in model:
#         return PassThroughDict(model,
#             loss_inputs=loss_inputs,
#             loss=_reduce('loss'),
#             loss_data=_reduce('loss_data'),
#             loss_regularizer=_reduce('loss_regularizer')
#         )
#     else:
#         return PassThroughDict(model,
#             loss_inputs=loss_inputs,
#             loss=_reduce('loss')
#         )



"""
Numericalize Postmaps
---------------------
"""


def flat_numericalize_postmap(model, flat_key="flat", mode=None,
                              annealing_combiner=None, mapreduce=None,
                              save_compiled_functions=True, initial_inputs=None, adapt_init_params=lambda ps: ps,
                              pre_compile_parameters_subgraph=False,
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

    parameters = model[flat_key]
    derivatives = {
        "num_loss": lambda loss: model[loss],
        "num_jacobian": lambda loss: theano.grad(model[loss], parameters, disconnected_inputs="warn"),
        "num_hessian": lambda loss: theano.gradient.hessian(model[loss], parameters)
    }
    general_theano_kwargs = dict(on_unused_input="ignore", allow_input_downcast=True, profile=profile, mode=mode)
    def theano_function(*args, **kwargs):
        update(kwargs, general_theano_kwargs, overwrite=False)
        return theano.function(*args, **kwargs)

    def function(outputs):
        """ compiles function with signature f(params, *loss_inputs) """
        if pre_compile_parameters_subgraph:
            # we are only interested in subgraph of parameters
            sub, _ = independent_subgraphs([parameters], model['loss_inputs'], outputs)
            # ``sub`` includes everything needed for computing outputs beside loss_inputs
            fparam = theano_function([parameters], sub)
            foutput = theano_function(sub + model['loss_inputs'], outputs)

            if mapreduce is not None:
                def f(parameters, *loss_inputs):
                    rparam = fparam(parameters)
                    def h(*inner_loss_inputs):
                        return foutput(*(rparam + list(inner_loss_inputs)))
                    return mapreduce(h, *loss_inputs)
            else:
                def f(parameters, *loss_inputs):
                    return foutput(*(fparam(parameters) + list(loss_inputs)))
            f.wrapped = fparam, foutput
        else:
            _f = theano_function([parameters] + model['loss_inputs'], outputs)
            if mapreduce is not None:
                def f(parameters, *loss_inputs):
                    def h(*inner_loss_inputs):
                        return _f(parameters, *inner_loss_inputs)
                    return mapreduce(h, *loss_inputs)
            else:
                f = _f
            f.wrapped = _f
        return f

    def numericalize(key):
        try:
            if annealing_combiner:
                return annealing_combiner(
                    function(derivatives[key]("loss_data")),
                    function(derivatives[key]("loss_regularizer"))
                )
            else:
                return function(derivatives[key]("loss"))
        except (KeyError, TypeError, ValueError) as e:
            raise KeyError("requested key %s not computable. Internal Error: %s" % (key, e))
        except AssertionError:
            # TODO got the following AssertionError which seems to be a bug deep in theano/proxifying theano
            # "Scan has returned a list of updates. This should not happen! Report this to theano-users (also include the script that generated the error)"
            # for now we ignore this
            raise KeyError("Internal Theano AssertionError. Hopefully, this will get fixed in the future.")

    num_parameters = theano.function(model['inputs'], parameters, on_unused_input='ignore')(*initial_inputs)

    dd = DefaultDict(  # DefaultDict will save keys after they are called the first time
        default_getitem=lambda key: numericalize(key) if key in derivatives else model[key],
        default_setitem=lambda key, value: NotImplementedError("You cannot set items on a numericalize postmap."),
        # if this would be noexpand always, we could do it savely, but without not
        num_parameters=adapt_init_params(num_parameters)
    )  # TODO add information about keys in derivatives into DefaultDict
    return dd if save_compiled_functions else dd.noexpand()


# this is not itself a postmap, however essential ans specific helper for the numericalize postmap

class AnnealingCombiner(object):
    """ linearly combines functions with given weights, where weights change over time
    """
    def __init__(self, weights=izip(repeat(1), repeat(1))):
        """
        Parameters
        ----------
        weights : list of infinite generators
            keyword-argument. Referring to respective weights, how to combine functions ``fs``
            ``len(weights) == len(fs)`` must hold and weights must refer to INFINITE generators,
            as looping makes no sense at all for annealing

            defaults to expecting only two functions and adds them up
        """
        self.weights = weights

    def __call__(self, *functions):
        """
        Parameters
        ----------
        functions : list of functions
            functions to be combined
        """
        assert len(self.weights) == len(functions), "there should be equal amount of weight lines and functions"
        def annealed(*args):
            s = 0
            for f, w in izip(functions, self.weights):
                s += next(w) * f(*args)
            return s
        return annealed


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