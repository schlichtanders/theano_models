#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import operator as op
import traceback
from collections import Sequence
from itertools import izip, repeat
import warnings

import numpy as np
import theano
import theano.tensor as T
from schlichtanders.mycontextmanagers import ignored
from schlichtanders.mylists import as_list, deflatten
from theano import gof
from schlichtanders.mydicts import PassThroughDict, DefaultDict, update
from schlichtanders.myfunctools import fmap, sumexpmap, convert, meanexpmap, sumexp, AverageExp, lift
from util import list_random_sources

from base import Model, Merge, outputting_references
from base_tools import norm_distance, L2
from theano.gof.fg import MissingInputError
from theano_models.util.theano_helpers import independent_subgraphs, graphopt_merge_add_mul, \
    independent_subgraphs_extend_add_mul, rebuild_graph
from util import clone_renew_rng
from theano.compile import SharedVariable, rebuild_collect_shared


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


def loss_deterministic(model, distance=norm_distance):
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
                 batch_common_rng=True, batch_mapreduce=None, batch_precompile=True,
                 exp_average_n=0,
                 exp_ratio_estimator=None,
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
    batch_precompile : bool
        precompilation (True) means, that everything related to the parameters only is precomputed per batch

    theano_function_kwargs : dict
        additional arguments passed to theano.function

    exp_average_n : int
        if an average in exponential space shall be done (i.e. in case of loglikelihood-loss on probability values)
        use this, an averaging is applied to each sample independently

    exp_ratio_estimator : str or None
        method used for ratio estimator in case of `exp_average_n > 1`. Used for estimating the gradient.
        'grouping', 'firstorder' or None

    Returns
    -------
    DefaultDict of num_loss, num_jacobian and num_hessian
    """

    # Parameter Preprocessing
    # -----------------------
    assert (not isinstance(parameters, Sequence)), "Currently only single theano expression for parameters is supported."
    if batch_mapreduce is None:
        # no precompile, if there is nothing to map upon
        batch_precompile = False
    elif exp_average_n > 1:
        # no warning, as batch_precompile=True is default value
        # if batch_precompile:
        #     warnings.warn("``average_exp=True`` requires ``batch_precompile=False``,"
        #                   "as the latter is reset to ``True`` (was False)")
        batch_precompile = False  # TODO add precompile = True version
        batch_common_rng = False  # for precompile = True this is needed, otherwise averaging wouldn't make sense
    elif batch_common_rng:
        # precompile of there is something to map, and if same random numbers shall be used in every sample
        if not batch_precompile:
            warnings.warn("``batch_common_rng=True`` and ``batch_mapreduce`` require ``batch_precompile=True``,"
                          "as the latter is reset to ``True`` (was False)")
        batch_precompile = True

    theano_function_kwargs['on_unused_input'] = "ignore"
    theano_function_kwargs['allow_input_downcast'] = True

    def theano_function(*args, **kwargs):
        update(kwargs, theano_function_kwargs, overwrite=False)
        return wrapper(theano.function(*args, **kwargs))

    # Core
    # ----
    derivatives = {
        "num_loss": lambda key: loss[key],
        "num_jacobian": lambda key: theano.grad(loss[key], parameters, disconnected_inputs="warn"),
        "num_hessian": lambda key: theano.gradient.hessian(loss[key], parameters)
    }

    def _numericalize(key_degree, key_loss):
        """ compiles function with signature f(num_params, *loss_inputs) """
        output = derivatives[key_degree](key_loss)
        if batch_precompile:
            print "batch_precompile"
            # further the subgraph ``sub`` is computed
            # it includes everything needed for separating parameters from outputs
            noise_source = list_random_sources(output)  # not dependend on anything
            inputs = [parameters] + loss['inputs'] + noise_source
            inputs, outputs, givens = rebuild_graph(inputs, [output], graphopt_merge_add_mul)
            _parameters, loss_inputs, noise_vars = deflatten(inputs, [parameters, loss['inputs'], noise_source])
            output = outputs[0]

            # find respective subgraphs
            if batch_common_rng:
                sub, _ = independent_subgraphs([_parameters] + noise_vars, loss_inputs, outputs)
                # extend subgraphs to include subparts of subsequent add/mul operators:
                sub = independent_subgraphs_extend_add_mul(sub)
                # build respective part-functions, applying full graph optimization on each
                fparam = theano_function([_parameters], sub, givens=givens)
                foutput = theano_function(sub + loss_inputs, output)
            else:
                sub, _ = independent_subgraphs([_parameters], loss_inputs + noise_vars, outputs)
                # extend subgraphs to include subparts of subsequent add/mul operators:
                sub = independent_subgraphs_extend_add_mul(sub)
                # build respective part-functions, applying full graph optimization on each
                fparam = theano_function([_parameters], sub)
                foutput = theano_function(sub + loss_inputs, output, givens=givens)

            def f(num_params, *loss_inputs):
                num_sub = fparam(num_params)
                def h(*inner_loss_inputs):
                    return foutput(*(num_sub + list(inner_loss_inputs)))
                return batch_mapreduce(h, *loss_inputs)
            f.wrapped = fparam, foutput

        else:
            if exp_average_n > 1:
                if key_degree == "num_loss":
                    _f = theano_function([parameters] + loss['inputs'], output)  # TODO the parameters part could be precompiled in principle
                    _f = lift(_f, AverageExp(exp_average_n))
                elif key_degree == "num_jacobian":
                    Output = derivatives["num_loss"](key_loss)  # stammfunktion, therefore capital

                    __f = theano.function([parameters] + loss['inputs'],
                                         [Output, output])  # TODO the parameters part could be precompiled in principle

                    def _f(num_params, *loss_inputs):
                        def logP_logDerivative():
                            for _ in xrange(exp_average_n):
                                yield __f(num_params, *loss_inputs)
                        log_Ps, log_derivatives = list(zip(*logP_logDerivative()))
                        log_Ps, log_derivatives = np.asarray(log_Ps), np.asarray(log_derivatives)
                        N = len(log_Ps)
                        assert N == len(log_derivatives), "this should be the same"

                        xs = np.exp(log_Ps)[:, None]
                        ys = log_derivatives * xs
                        if exp_ratio_estimator is None:
                            # the biased estimator:
                            return ys.sum(0) / xs.sum(0)
                        elif exp_ratio_estimator == "grouping":
                            # grouping formula for ratio estimator (applied to each gradient entry separately):
                            return N * ys.sum(0) / xs.sum(0) - (N - 1) / N * log_derivatives.sum(0)  # this worked with certain initial random variables
                        elif exp_ratio_estimator == "firstorder":
                            # linear formula for ratio estimator: (does not seem to work, makes variance small again)
                            xs_sum = xs.sum(0)
                            xs_mean = xs_sum / N
                            _cov = (xs - xs_mean) * (log_derivatives - log_derivatives.mean(0))
                            return ys.sum(0)/xs_sum - _cov.mean(0)/xs_mean
                else:
                    raise KeyError("key %s is not supported with `n_average_exp > 1`" % key_degree)

            else:
                _f = theano_function([parameters] + loss['inputs'], output)

            if batch_mapreduce is not None:
                def f(num_params, *loss_inputs):
                    def h(*inner_loss_inputs):
                        return _f(num_params, *inner_loss_inputs)
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
                    _numericalize(key, "loss_data"),
                    theano_function([parameters], derivatives[key]("loss_regularizer"))
                )
            else:
                return _numericalize(key, "outputs")

        except (KeyError, TypeError, ValueError) as e:
            raise
            raise KeyError("Key '%s' not computable. Internal Error: %s" % (key, e))
        except AssertionError as e:
            # TODO got the following AssertionError which seems to be a bug deep in theano/proxifying theano
            # "Scan has returned a list of updates. This should not happen! Report this to theano-users (also include the script that generated the error)"
            # for now we ignore this
            raise
            raise KeyError("Internal Theano AssertionError. Hopefully, this will get fixed in the future. Internal Error: %s" % traceback.format_exc())

    return DefaultDict(  # DefaultDict will save keys after they are called the first time
        default_getitem=get_numericalized,
        default_setitem=lambda key, value: NotImplementedError("You cannot set items on a numericalize postmap."),
        # if this would be noexpand always, we could do it savely, but without not
        num_parameters=adapt_init_params(parameters.eval(initial_givens))
    )  # TODO add information about keys in derivatives into DefaultDict


'''
# TODO test numericalizeExp !!!
def numericalizeExp(loss, parameters,
                 annealing_combiner=None, wrapper=lambda f: f,
                 initial_givens={}, adapt_init_params=lambda ps: ps,
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

    theano_function_kwargs['on_unused_input'] = "ignore"
    theano_function_kwargs['allow_input_downcast'] = True

    def theano_function(*args, **kwargs):
        update(kwargs, theano_function_kwargs, overwrite=False)
        return wrapper(theano.function(*args, **kwargs))

    def _numericalize(loss_key):
        ret = {}
        # loss
        # ----
        noutput = - loss[loss_key]  # minus is important for meanexpmap, negative output

        noise_source = list_random_sources(noutput)  # not dependend on anything
        inputs = [parameters] + loss['inputs'] + noise_source
        inputs, noutputs, ngivens = rebuild_graph(inputs, [noutput], graphopt_merge_add_mul)
        _parameters, loss_inputs, noise_vars = deflatten(inputs, [parameters, loss['inputs'], noise_source])
        noutput = noutputs[0]

        # we need distinct random numbers everywhere for the ratio estimator:
        nsub, _ = independent_subgraphs([_parameters], loss_inputs + noise_vars, noutput)
        nsub = independent_subgraphs_extend_add_mul(nsub)

        nfparam = theano_function([_parameters], nsub)  # negative function parameter part
        nfoutput = theano_function(nsub + loss_inputs, noutput, givens=ngivens)  # negative function output part

        def f(num_params, *loss_inputs):
            nnum_sub = nfparam(num_params)
            def h(*inner_loss_inputs):
                return nfoutput(*(nnum_sub + list(inner_loss_inputs)))  # returns logP
            return - meanexpmap(h, *loss_inputs)  # mirrows the minus in the beginning
        f.wrapped = nfparam, nfoutput
        ret['num_loss'] = f

        # derivative:
        # -----------
        # NOTE: we cannot reuse the above theano expressions, as graph optimization may lead non gradiable nodes

        doutput = theano.grad(loss[loss_key], parameters, disconnected_inputs="warn")  # -output = loss[loss_key] (with add/mul merged)
        noise_source = list_random_sources(doutput)  # not dependend on anything  # should be the same as above
        inputs = [parameters] + loss['inputs'] + noise_source
        inputs, doutputs, dgivens = rebuild_graph(inputs, [doutput], graphopt_merge_add_mul)
        _parameters, loss_inputs, noise_vars = deflatten(inputs, [parameters, loss['inputs'], noise_vars])
        doutput = doutputs[0]

        # we need distinct random numbers everywhere for the ratio estimator:
        dsub, _ = independent_subgraphs([_parameters], loss_inputs + noise_vars, doutput)
        dsub = independent_subgraphs_extend_add_mul(dsub)
        dfparam = theano_function([_parameters], dsub)
        dfoutput = theano_function(dsub + loss_inputs, doutput, givens=dgivens)

        def df(num_params, *loss_inputs):
            def fix_type(x):
                if hasattr(x, 'next'):  # no generators, as we need the information twice
                    return list(x)
                if hasattr(x, '__iter__'):
                    return x
                raise RuntimeError("This should not happen. Iterable types are expected as loss_inputs.")
            loss_inputs = map(fix_type, loss_inputs)  # we use this 2 times, i.e. generators are not useful
            nnum_sub = nfparam(num_params)
            def h_logP(*inner_loss_inputs):
                return nfoutput(*(nnum_sub + list(inner_loss_inputs)))
            log_Ps = np.asarray(map(h_logP, *loss_inputs))

            dnum_sub = dfparam(num_params)
            def h_derv(*inner_loss_inputs):
                return dfoutput(*(dnum_sub + list(inner_loss_inputs)))
            log_derivatives = np.asarray(map(h_derv, *loss_inputs))
            N = len(log_Ps)
            assert N == len(log_derivatives), "this should be the same"

            xs = np.exp(log_Ps)[:, None]
            ys = log_derivatives * xs
            # grouping formula for ratio estimator:
            # return N * ys.sum() / xs.sum() - (N-1)/N * log_derivatives.sum()  # this worked with certain initial random variables
            return ys.sum() / xs.sum()
            # linear formula for ratio estimator: (does not seem to work, makes variance small again)
            # xs_sum = xs.sum()
            # xs_mean = xs_sum / N
            # _cov = (xs - xs_mean) * (log_derivatives - log_derivatives.mean(axis=0))
            # return ys.sum()/xs_sum - _cov.mean(axis=0)/xs_mean

        df.wrapped = dfparam, dfoutput
        ret['num_jacobian'] = df
        return ret

    if annealing_combiner:
        numericalized = _numericalize("loss_data")
        numericalized['num_loss'] = annealing_combiner(
            numericalized['num_loss'],
            theano_function([parameters], loss["loss_regularizer"])
        )
        numericalized['num_jacobian'] = annealing_combiner(
            numericalized['num_jacobian'],
            theano_function([parameters], theano.grad(loss["loss_regularizer"], parameters, disconnected_inputs="warn"))
        )
    else:
        numericalized = _numericalize("outputs")

    numericalized['num_parameters'] = adapt_init_params(parameters.eval(initial_givens))
    return numericalized
'''

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