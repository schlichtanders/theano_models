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

from base import Model, Merge, outputting_references
from base_tools import norm_distance, L2
from theano.gof.fg import MissingInputError
from theano_models.util.theano_helpers import independent_subgraphs
from util import clone_renew_rng


__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Symbolic Loss
=============

Generating theano loss expressions.
"""


"""
Standard Loss
-------------
"""


def loss_deterministic(model, distance=norm_distance()):
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
    assigns and returns model.loss
    """
    if isinstance(model['outputs'], gof.graph.Variable):
        targets = [model['outputs'].type("deterministic_target")]
        outputs = [model['outputs']]
    else:
        targets = [o.type() for o in model['outputs']]
        outputs = model['outputs']

    model.loss = Merge(model, name=model.name + ".loss", ignore_references=outputting_references,
        inputs=targets + model['inputs'],
        outputs=distance(targets, outputs)
    )
    return model.loss


def loss_probabilistic(model):
    """ builds premap for a standard probabilistic model

    Parameters
    ----------
    model : Model
        to be transformed

    Returns
    -------
    assigns and returns model.loss
    """
    model.loss = Merge(model, name=model.name + ".loss", ignore_references=outputting_references,
        inputs=model.logP['inputs'] + model['inputs'],
        outputs=-model.logP['outputs']
    )
    return model.loss


"""
Regularized/Annealed Loss
-------------------------

Adding a regularizer is a common procedure to improve generalizability of the model (which is usually what we want).
The following define such loss functions.
"""


def loss_regularizer(model, regularizer_norm=L2, regularizer_scalar=1, key_parameters="flat"):
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
    assigns and returns model.loss
    """
    if not hasattr(model, 'loss'):
        raise ValueError("need model with loss to add regularizer")
    loss_data = model.loss['outputs']
    loss_regularizer = regularizer_norm(model[key_parameters])
    model.loss = Merge(model, name=model.name + ".loss_regularizer", ignore_references=outputting_references,
        inputs=model.loss['inputs'],
        loss_data=loss_data,
        loss_regularizer=loss_regularizer,
        outputs=loss_data + regularizer_scalar * loss_regularizer
    )
    return model.loss


def loss_variational(model):
    """use this postmap INSTEAD of the standard probabilistic postmap

    Returns
    -------
    assigns and returns model.loss
    """
    model.loss = Merge(model, name=model.name+".loss_variational", ignore_references=outputting_references,
        inputs=model.logP['inputs'] + model['inputs'],
        outputs=-model.logP['outputs'],
        loss_data=-model.loglikelihood['outputs'],  # indeed uses the same inputs as model.logP
        loss_regularizer=model['kl_prior']
    )
    return model.loss


def loss_normalizingflow(model):
    """ use this postmap INSTEAD of the standard probabilistic postmap

    Returns
    -------
    assigns and returns model.loss
    """
    model.loss = Merge(model, name=model.name+".loss_normflow", ignore_references=outputting_references,
        inputs=model.logP['inputs'] + model['inputs'],
        outputs=-model.logP['outputs'],
        loss_data=-model.loglikelihood['outputs'] - model['logprior'],
        loss_regularizer=model['logposterior'],
    )
    return model.loss


"""
Numerical Loss
==============

generating numerical loss functions
"""


# TODO numericalize using maximum likelihood ideas
# https://books.google.de/books?id=ObSs1O9X30AC&pg=PA182&lpg=PA182&dq=markov+chain+monte+carlo+for+%22ratio+estimator%22&source=bl&ots=wOYqD6ZMLQ&sig=__1LiCBZq4kVD25syhAB4e7XVn8&hl=en&sa=X&ved=0ahUKEwiUp9_DvN_NAhUL7hoKHfBRAF8Q6AEIIDAB#v=onepage&q=markov%20chain%20monte%20carlo%20for%20%22ratio%20estimator%22&f=false

def numericalize(loss, parameters,
                 annealing_combiner=None, wrapper=lambda f:f,
                 initial_givens={}, adapt_init_params=lambda ps: ps,
                 batch_common_rng=True, batch_mapreduce=None, batch_precompile=None,
                 **theano_function_kwargs):
    """ Produces interface for standard numerical optimizer

    Parameters
    ----------
    loss : Model
        loss expressions to be numericalized
    parameters : theano expression
        single theano expression which denotes all parameters of the model which shall be optimized

    annealing_combiner : None or AnnealingCombiner
        indicating whether 'loss_data' and 'loss_regularizer' should be used (annealing=True) or 'loss' (default)
    wrapper : function f -> f where f function like used in scipy.optimize.minimize
        mainly intented for adding possibility to Average
        is applied to any theano.function which is created (i.e. loss or derivative, loss_data or loss_regularizer)

    initial_givens : dict TheanoVariable -> TheanoVariable
        givens/replace dictionary for creating num_parameters
        e.g. for parameters which are not grounded, but depend on a non-grounded input
        (only needed for initialization)
    adapt_init_params : function numpy-vector -> numpy-vector
        for further control of initial parameters

    batch_mapreduce : fmap
        mapreduce which is applied over the data
    batch_common_rng : bool
        indicating whether same random number shall be used for the whole batch (True)
        or that random number generator shall be updated for each sample separately (False)
    batch_precompile : dict (key: bool)
        key must correspond to output key
        precompilation means, that everything related to the parameters only is precomputed per batch

    theano_function_kwargs : dict
        additional arguments passed to theano.function

    Returns
    -------
    DefaultDict over model
    """
    assert (not isinstance(parameters, Sequence)), "Currently only single theano expression for parameters is supported."
    if batch_precompile is None:
        batch_precompile = {'num_loss': True, 'num_jacobian': False, 'num_hessian': False}
    else:
        for key, v in {'num_loss': True, 'num_jacobian': False, 'num_hessian': False}.iteritems():
            if key not in batch_precompile:
                batch_precompile[key] = v

    derivatives = {
        "num_loss": lambda key: loss[key],
        "num_jacobian": lambda key: theano.grad(loss[key], parameters, disconnected_inputs="warn"),
        "num_hessian": lambda key: theano.gradient.hessian(loss[key], parameters)
    }

    theano_function_kwargs['on_unused_input'] = "ignore"
    theano_function_kwargs['allow_input_downcast'] = True
    def theano_function(*args, **kwargs):
        update(kwargs, theano_function_kwargs, overwrite=False)
        return wrapper(theano.function(*args, **kwargs))

    def _numericalize(outputs, precompile):
        """ compiles function with signature f(params, *loss_inputs) """

        # error prone, therefore deprecated for now
        '''
        if pre_compile == "build_batch_theano_graph":  # batch_size != None must be ensured (see ValueError above)
            # TODO this pattern seems to be useful very very often, however compilation time is almost infinite (felt like that)
            # TODO ask on theano, whether this pattern can be made more efficient
            # build new loss_inputs with extra dimension (will be regarded as first dimension)
            batch_loss_inputs = [T.TensorType(i.dtype, i.broadcastable + (False,))(i.name + ("" if i.name is None else "R"))
                                for i in model['loss_inputs']]
            def clones():
                for i in xrange(batch_size):
                    yield clone_renew_rng(outputs, replace=dict(izip(model['loss_inputs'], [a[i] for a in batch_loss_inputs])))
            batch_outputs = T.add(*clones())
            f = theano_function([parameters] + batch_loss_inputs, batch_outputs)
        '''
        if (precompile and batch_mapreduce is not None) or (batch_common_rng and batch_mapreduce is not None):
            # we need to handle randomness per sample
            # this is  not used for now and not for the ideal case, hence deprecated
            '''
            # using model['noise'] is confusing when using another rng in the background, as then the randomness occurs
            # before and hence can go into ``sub``
            # therefore we always search for rng automatically
            if pre_compile == "use_compiled_functions":
                singleton = not isinstance(outputs, Sequence)
                _f = theano_function([parameters] + loss['inputs'], outputs)
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
            '''
            # further the subgraph ``sub`` is computed
            # it includes everything needed for separating parameters from outputs
            noise_source = list_random_sources(outputs)  # not dependend on anything
            if batch_common_rng:
                sub, _ = independent_subgraphs([parameters] + noise_source, loss['inputs'], outputs)
            else:
                sub, _ = independent_subgraphs([parameters], loss['inputs'] + noise_source, outputs)
            fparam = theano_function([parameters], sub)
            foutput = theano_function(sub + loss['inputs'], outputs)

            def f(parameters, *loss_inputs):
                rparam = fparam(parameters)
                def h(*inner_loss_inputs):
                    return foutput(*(rparam + list(inner_loss_inputs)))
                return batch_mapreduce(h, *loss_inputs)
            f.wrapped = fparam, foutput

        else:  # this is to be erased as soon as pre_compilation is optimized
            _f = theano_function([parameters] + loss['inputs'], outputs)
            if batch_mapreduce is not None:
                def f(parameters, *loss_inputs):
                    def h(*inner_loss_inputs):
                        return _f(parameters, *inner_loss_inputs)
                    return batch_mapreduce(h, *loss_inputs)
                f.wrapped = _f
            else:
                f = _f
        return f

    def get_numericalized(key):
        if key not in derivatives:
            raise KeyError("Key '%s' not computable." % key)
        try:
            if annealing_combiner:
                return annealing_combiner(
                    _numericalize(derivatives[key]("loss_data"), batch_precompile[key]),
                    theano_function([parameters], derivatives[key]("loss_regularizer"))
                )
            else:
                return _numericalize(derivatives[key]("outputs"), batch_precompile[key])

        except (KeyError, TypeError, ValueError) as e:
            raise KeyError("Key '%s' not computable. Internal Error: %s" % (key, e))
        except AssertionError:
            # TODO got the following AssertionError which seems to be a bug deep in theano/proxifying theano
            # "Scan has returned a list of updates. This should not happen! Report this to theano-users (also include the script that generated the error)"
            # for now we ignore this
            raise KeyError("Internal Theano AssertionError. Hopefully, this will get fixed in the future.")

    dd = DefaultDict(  # DefaultDict will save keys after they are called the first time
        default_getitem=get_numericalized,
        default_setitem=lambda key, value: NotImplementedError("You cannot set items on a numericalize postmap."),
        # if this would be noexpand always, we could do it savely, but without not
        num_parameters=adapt_init_params(parameters.eval(initial_givens))
    )  # TODO add information about keys in derivatives into DefaultDict
    return dd


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


def scipy_kwargs(numerical):
    """ extracts kwargs for scipy.optimize.minize as far as available and mandatory

    Parameters
    ----------
    numerical: dict-like
        with numerical functions for optimization

    Returns
    -------
    dict
    """
    kwargs = {
        "fun": numerical["num_loss"],
        "x0": numerical["num_parameters"],
    }
    with ignored(KeyError):
        kwargs["jac"] = numerical["num_jacobian"]

    try:
        kwargs["hessp"] = numerical["num_hessianp"]
    except KeyError:
        with ignored(KeyError):
            kwargs["hess"] = numerical["num_hessian"]
    return kwargs


def climin_kwargs(numerical):
    """ extracts kwargs for climin.util.optimizer as far as available and mandatory

    Parameters
    ----------
    numerical: dict-like
        with numerical functions for optimization
    Returns
    -------
    dict
    """
    kwargs = {
        "f": numerical["num_loss"],
        "wrt": numerical["num_parameters"],
    }
    with ignored(KeyError):
        kwargs["fprime"] = numerical["num_jacobian"]
    return kwargs