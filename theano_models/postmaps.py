#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import operator as op
from collections import Sequence
from itertools import izip, repeat

import numpy as np
import theano
import theano.tensor as T
from schlichtanders.mycontextmanagers import ignored
from schlichtanders.mylists import as_list
from theano import gof
from schlichtanders.mydicts import PassThroughDict, DefaultDict, update
from schlichtanders.myfunctools import fmap
from util import list_random_sources

from base_tools import norm_distance, L2
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
    return PassThroughDict(model,
        loss_inputs=model.logP['inputs'] + model['inputs'],
        loss=-model.logP['outputs']
    )


"""
Annealing Postmaps
------------------

Adding a regularizer is a common procedure to improve generalizability of the model (which is usually what we want).
The following define such postmaps.
"""


def regularizer_postmap(model, regularizer_norm=L2, regularizer_scalar=1, key_parameters="flat"):
    """ postmap for a standard deterministic model. Simply add this postmap to the model.

    Parameters
    ----------
    model : Model
        kwargs to be adapted by this postmap
    regularizer_norm : function working on list of parameters, returning scalar loss
        shall regularize parameters
    regularizer_scalar : scalar
        weight of loss_regularizer (loss_data is weighted with 1)
    key_parameters : str
        which reference should be counted as the parameters (and regularized)

    Returns
    -------
    IdentityDict over model
    """
    return PassThroughDict(model,
        loss_data=model['loss'],
        loss_regularizer=regularizer_norm(model[key_parameters]),
        loss=model['loss'] + regularizer_scalar*regularizer_norm(model[key_parameters])
    )


def variational_postmap(model):
    """use this postmap INSTEAD of the standard probabilistic postmap"""
    return PassThroughDict(model,
        loss_inputs=model.logP['inputs'] + model['inputs'],
        loss=-model.logP['outputs'],
        loss_data=-model.loglikelihood['outputs'],  # indeed uses the same inputs as model.logP
        loss_regularizer=model['kl_prior']
    )


def normalizingflow_postmap(model):
    """ use this postmap INSTEAD of the standard probabilistic postmap """
    return PassThroughDict(model,
        loss_inputs=model.logP['inputs'] + model['inputs'],
        loss=-model.logP['outputs'],
        loss_data=-model.loglikelihood['outputs'] - model['logprior'],
        loss_regularizer=model['logposterior'],
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
                              annealing_combiner=None, mapreduce=None, wrapper=lambda f:f,
                              save_compiled_functions=True, initial_givens={}, adapt_init_params=lambda ps: ps,
                              pre_compile=None,
                              batch_size=None,
                              profile=False):
    """ postmap to offer an interface for standard numerical optimizer

    'loss' and etc. must be available in the model

    Parameters
    ----------
    model : Model
    annealing_combiner : None or AnnealingCombiner
        indicating whether 'loss_data' and 'loss_regularizer' should be used (annealing=True) or 'loss' (default)
    mapreduce : fmap
        like wrapper, however this is only applied to loss_data, therefore the naming mapreduce
    wrapper : function f -> f where f function like used in scipy.optimize.minimize
        mainly intented for adding possibility to Average
        is applied to any theano.function which is created (i.e. both data and regularizer for instance)
    save_compiled_functions : bool
        If false, functions are compiled on every postmap call anew. If true, they are hashed like in a usual DefaultDict
    initial_givens : dict TheanoVariable -> TheanoVariable
        givens/replace dictionary for creating num_parameters
        e.g. for parameters which are not grounded, but depend on a non-grounded input
        (only needed for initialization)
    adapt_init_params : function numpy-vector -> numpy-vector
        for further control of initial parameters
    pre_compile : dictionary of pre_compile flags for the function compilation
        default is chosen such that in the test case the performance was optimal (both in compile time and runtime)
        special keys::
            False - no pre_compilation at all
            True - pre_compilation on model itself
            'use_compiled_functions' - like True, but with pre_compilation based on the optimized graph
            'build_batch_theano_graph' - only for very small batch sizes. Computes a theanograph for everything
    batch_size : int
        needed for pre_compile option 'build_batch_theano_graph' (however not recommended for large batch_sizes)

    Returns
    -------
    DefaultDict over model
    """
    assert (not isinstance(model[flat_key], Sequence)), "need single flat vector"
    if pre_compile is None:
        pre_compile = {'num_loss': True, 'num_jacobian': False, 'num_hessian': False}
    else:
        for key, v in {'num_loss': True, 'num_jacobian': False, 'num_hessian': False}.iteritems():
            if key not in pre_compile:
                pre_compile[key] = v

    parameters = model[flat_key]
    derivatives = {
        "num_loss": lambda loss: model[loss],
        "num_jacobian": lambda loss: theano.grad(model[loss], parameters, disconnected_inputs="warn"),
        "num_hessian": lambda loss: theano.gradient.hessian(model[loss], parameters)
    }
    general_theano_kwargs = dict(on_unused_input="ignore", allow_input_downcast=True, profile=profile, mode=mode)
    def theano_function(*args, **kwargs):
        update(kwargs, general_theano_kwargs, overwrite=False)
        return wrapper(theano.function(*args, **kwargs))

    def function(outputs, pre_compile):
        """ compiles function with signature f(params, *loss_inputs) """
        # error prone, therefore deprecated for now
        # if pre_compile == "build_batch_theano_graph":  # batch_size != None must be ensured (see ValueError above)
        #     # TODO this pattern seems to be useful very very often, however compilation time is almost infinite (felt like that)
        #     # TODO ask on theano, whether this pattern can be made more efficient
        #     # build new loss_inputs with extra dimension (will be regarded as first dimension)
        #     batch_loss_inputs = [T.TensorType(i.dtype, i.broadcastable + (False,))(i.name + ("" if i.name is None else "R"))
        #                         for i in model['loss_inputs']]
        #     def clones():
        #         for i in xrange(batch_size):
        #             yield clone_renew_rng(outputs, replace=dict(izip(model['loss_inputs'], [a[i] for a in batch_loss_inputs])))
        #     batch_outputs = T.add(*clones())
        #     f = theano_function([parameters] + batch_loss_inputs, batch_outputs)

        if pre_compile and mapreduce is not None:
            # we need to handle randomness per sample
            # using model['noise'] is confusing when using another rng in the background, as then the randomness occurs
            # before and hence can go into ``sub``
            # therefore we always search for rng automatically
            if pre_compile == "use_compiled_functions":
                singleton = not isinstance(outputs, Sequence)
                _f = theano_function([parameters] + model['loss_inputs'], outputs)
                noise_source = []
                for i, s in izip(_f.maker.inputs, _f.input_storage):
                    if s.data is not None:
                        if str(i.variable) == '<RandomStateType>':
                            noise_source.append(i.variable)
                outputs = [o.variable for o in _f.maker.outputs]
                if singleton:
                    outputs = outputs[0]
            else:
                # standard precompile version, note that this can be significantly slower than without using any precompile
                noise_source = list_random_sources(outputs)

            # further the subgraph ``sub`` is computed
            # it includes everything needed for separating parameters from outputs
            sub, _ = independent_subgraphs([parameters], model['loss_inputs'] + noise_source, outputs)
            fparam = theano_function([parameters], sub)
            foutput = theano_function(sub + model['loss_inputs'], outputs)

            def f(parameters, *loss_inputs):
                rparam = fparam(parameters)
                def h(*inner_loss_inputs):
                    return foutput(*(rparam + list(inner_loss_inputs)))
                return mapreduce(h, *loss_inputs)
            f.wrapped = fparam, foutput

        else:
            _f = theano_function([parameters] + model['loss_inputs'], outputs)
            if mapreduce is not None:
                def f(parameters, *loss_inputs):
                    def h(*inner_loss_inputs):
                        return _f(parameters, *inner_loss_inputs)
                    return mapreduce(h, *loss_inputs)
                f.wrapped = _f
            else:
                f = _f
        return f

    def numericalize(key):
        try:
            if annealing_combiner:
                return annealing_combiner(
                    function(derivatives[key]("loss_data"), pre_compile[key]),
                    theano_function([parameters], derivatives[key]("loss_regularizer"))
                )
            else:
                return function(derivatives[key]("loss"), pre_compile[key])
        except (KeyError, TypeError, ValueError) as e:
            raise KeyError("requested key %s not computable. Internal Error: %s" % (key, e))
        except AssertionError:
            # TODO got the following AssertionError which seems to be a bug deep in theano/proxifying theano
            # "Scan has returned a list of updates. This should not happen! Report this to theano-users (also include the script that generated the error)"
            # for now we ignore this
            raise KeyError("Internal Theano AssertionError. Hopefully, this will get fixed in the future.")

    # num_parameters = theano.function(model['inputs'], parameters, on_unused_input='ignore')(*initial_inputs)
    num_parameters = parameters.eval(initial_givens)

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
    def __init__(self, weights_data=repeat(1), weights_regularizer=repeat(1), scale_by_len=("data", "regularizer")):
        """
        Parameters
        ----------
        weights_data : infinite generators
            consecutive weights for loss_data, defaults to constant 1
        weights_regularizer : infinte generator
            consecutive weights for loss_regularizer, defaults to constant 1
        """
        self.weights_data = weights_data
        self.weights_regularizer = weights_regularizer
        self.scale_by_len = scale_by_len

    def __call__(self, loss_data, loss_regularizer):
        """
        Parameters
        ----------
        loss_data: scalar function
            should have signature loss_data(parameters, *loss_inputs)
        loss_regularizer: scalar function which returns support + and *
            should have signature loss_regularizer(parameters)
        no_annealing : bool
            for extra executions of the functions which shall not advance the weights
        """
        def annealed(parameters, *loss_inputs, **kwargs):
            length = {'data': 1, 'regularizer': 1}
            with ignored(TypeError):
                for key in self.scale_by_len:
                    length[key] = len(loss_inputs[0])

            if kwargs.pop('no_annealing', False):
                return loss_data(parameters, *loss_inputs)/length['data'] + loss_regularizer(parameters)/length['regularizer']

            ld = next(self.weights_data) * loss_data(parameters, *loss_inputs) / length['data']
            lr = next(self.weights_regularizer) * loss_regularizer(parameters) / length['regularizer']
            return ld + lr
        annealed.wrapped = loss_data, loss_regularizer
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