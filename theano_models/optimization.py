#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

from itertools import izip

import numpy as np
import theano
import scipy.optimize
import climin.util

from schlichtanders.mydicts import update, IdentityDict
from schlichtanders.myfunctools import compose  # keep this for documentation reference
from schlichtanders.mylists import deepflatten
from schlichtanders.mynumpy import complex_reshape
from schlichtanders.myoptimizers import batch, online, average, annealing

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Generic Theano Optimization
===========================

This module offers an interface between arbitrary optimizers and Theano ``Graph`` objects.
To make a Graph optimizable is as simple as supporting certain references to Theano expression.
This set of keys not strict, but can be easily replaced by another one.

The main part of the interfacing work is done in wrapping an existent optimizer into a ``TheanoOptimizer``. The module
already implements generic TheanoOptimizers for Scipy and Climin, namely ``ScipyOptimizer``, ``ScipyAnnealingOptimizer``,
``CliminOptimizer``, and ``CliminAnnealingOptimizer``.
See below for more details how to construct a new ``TheanoOptimizer``.

If you want the optimizer dict-interface to look different than the Graph dict-interface (e.g. useful for Supervised
Models), you can specify a ``__optimizer_view__`` attribute which will be used instead of the graph.
See ``TheanoOptimizer.optimize`` for more details.
"""


class TheanoOptimizer(object):
    """ Lifts an arbitrary optimizer to work with Theano Graphs

    The essential parameter is ``kwargs_from_graph_remap`` dictionary.
    If you want to have a dynamic control over this parameter, subclass the TheanoOptimizer and make
    ``kwargs_from_graph`` a class method with one argument 'graph'.
    """

    def __init__(self, generic_optimizer_func, kwargs_from_graph, **kwargs):
        """ Initializes optimizer to work with Theano Graphs

        Parameters
        ----------
        generic_optimizer_func: function
            generic optimizer function like e.g. ``scipy.optimizer.minimize``
        kwargs_from_graph: function
            gets called with a Graph instance (usually OptimizableGraph)
            shall return a respective kwarg for ``generic_optimizer_func``

        kwargs
            additional kwargs for ``generic_optimizer_func``
        """
        self.generic_optimizer_func = generic_optimizer_func
        self.kwargs_from_graph = kwargs_from_graph
        self.kwargs = kwargs

    def set_kwargs(self, **kwargs):
        """ kwargs for the generic optimizer """
        self.kwargs.update(kwargs)

    def optimize(self, graph, **kwargs):
        """ calls this optimizer on the Graph (only the dictionary interface is needed)

        i.e. combines all (abstract) kwargs and calls ``self.generic_optimizer_func(**all_combined_kwargs)``

        If you have a Theano Graph with non-default references, you can pass instead a remappped version which remaps
        everything to the correct keys. The optimizer only needs a dict-like interface.
            >>> graph_view = remap(graph)

        You might want to specify a few extra remappings only, without shadowing remaining keys. For this use
        e.g. a ``schlichtanders.mydicts.IdentityDict`` as a ``remap`` in the above example.

        A graph can also define a class-wide default remapping (e.g. useful for Supervised Models where the
        optimizer needs different inputs than the model) by supplying a remap on ``graph.postmap`` (this defaults
        to the identity).

        Parameters
        ----------
        graph: Graph, OptimizableGraph
            graph to be optimized, usually of instance OptimizableGraph, however other Graphs work similarly
        kwargs:
            additional arguments for the optimizer function
        """
        kwargs_from_graph = self.kwargs_from_graph(graph.postmap())

        update(kwargs, kwargs_from_graph, overwrite=False)
        update(kwargs, self.kwargs, overwrite=False)
        return self.generic_optimizer_func(**kwargs)


"""
Creating New TheanoOptimizers
=============================

To lift an arbitrary optimizer to an optimizer which can work on Theano Graphs, the only extra which is needed is the
``kwargs_from_graph_remap`` dictionary.
It essentially maps a name to functions which is called with a Graph and must return the respective argument for the
optimizer. E.g. it must construct a loss function the optimizer can handle from the given theano expressions.


Abstract Remappings
-------------------
Abstract factory functions to interface Graphs conveniently.
"""


def numerical_parameters(graph):
    """ standard remap for accessing shared theano parameters

    if graph['parameters'] refers to singleton, then its numerical value is used directly, otherwise the numerical
    values of all parameters are flattened out and concatinated to give a numerical representation alltogether
    """
    num_parameters = []
    # singleton case:
    if len(graph['parameters']) == 1:
        # borrow=True is for 1) the case that the optimizer works inplace, 2) its faster
        return graph['parameters'][0].get_value(borrow=True)  # return it directly, without packing it into a list
    # else, flatten parameters out (as we cannot guarantee matching shapes):
    else:
        for p in graph['parameters']:
            v = p.get_value(borrow=True)  # p.get_value(borrow=True) # TODO what does borrow?
            num_parameters += deepflatten(v)
    return np.array(num_parameters)  # default to numpy type, as this supports numeric operators like indented


def numericalize(graph, loss_reference_name, d_order=0):
    """ numericalizes ``graph[loss_reference_name]`` or the respective derivative of order ``d_order``

    It works analogously to ``numerical_parameters`` in that it handles singleton cases or otherwise reshapes flattened
    numerical parameters.

    Parameters
    ----------
    graph : Graph
        source from which to wrap
    loss_reference_name: str
        graph[reference_name] will be wrapped
    d_order : int
        order of derivative (0 stands for no derivative) which shall be computed
    """
    # handle singleton parameters:
    parameters = graph['parameters'][0] if len(graph['parameters']) == 1 else graph['parameters']
    if d_order == 0:
        outputs = graph[loss_reference_name]
    elif d_order == 1:
        outputs = theano.grad(graph[loss_reference_name], parameters)
    elif d_order == 2:
        outputs = theano.gradient.hessian(graph[loss_reference_name], parameters)
    else:
        raise ValueError("Derivative of order %s is not yet implemented" % d_order)

    f_theano = theano.function(graph['loss_inputs'], outputs)
    shapes = [p.get_value(borrow=True).shape for p in graph['parameters']]  # borrow=True as it is faster

    def f(xs, *args, **kwargs):
        if len(graph['parameters']) == 1:
            graph['parameters'][0].set_value(xs, borrow=True)  # xs is not packed within list
            return f_theano(*args, **kwargs)  # where initialized correctly

        # else, reshape flattened parameters
        else:
            xs = list(complex_reshape(xs, shapes))
            for x, p in izip(xs, graph['parameters']):
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


"""
Basic example use with Scipy/Climin
-----------------------------------

As an example, here the basic way how to lift a generic optimizer function in order to work on Graphs.
We interface a standard OptimizableGraph.
"""

# TODO support missing gradient (e.g. Uniform distribution has no derivative)

def _scipy_remap(graph):
    return {
        "fun" : numericalize(graph, 'loss'),
        "jac" : numericalize(graph, 'loss', d_order=1),
        "x0"  : numerical_parameters(graph),
    }


def _climin_remap(graph):
    return {
        "f"      : numericalize(graph, 'loss'),
        "fprime" : numericalize(graph, 'loss', d_order=1),
        "wrt"    : numerical_parameters(graph),
    }


_scipy_optimizer = TheanoOptimizer(
    scipy.optimize.minimize,
    _scipy_remap
)

_climin_optimizer = TheanoOptimizer(
    climin.util.optimizer,
    _climin_remap,
)


"""
Subclassing TheanoOptimizer
---------------------------

For more dynamic fine control of the ``kwargs_from_graph_remap`` argument, it is very useful to simply subclass
``TheanoOptimizer``.

Such a generic dynamic interfaces to scipy.optimize.minimize and climin.util.optimizer is implemented in the following.
"""


class ScipyOptimizer(TheanoOptimizer):

    def __init__(self, wrapper=(lambda f, **wrapper_kwargs: f), **wrapper_kwargs):
        """
        Constructs a generic scipy optimizer, defaulting to no wrapper

        For mini-batch version with averaging use e.g. ``wrapper=compose(online, batch, average)``

        For setting wrapper_kwargs dnymically, just do for instance:
            >>> myoptimizer.wrapper_kwargs['repeat_n_times'] = 4

        Parameters
        ----------
        wrapper: higher level function
            wrapper to manipulate standard translations
        wrapper_kwargs: dict
            kwargs for the wrapper, e.g. for setting repeat_n_times
        """
        self.wrapper = wrapper
        self.wrapper_kwargs = wrapper_kwargs
        super(ScipyOptimizer, self).__init__(scipy.optimize.minimize, self.kwargs_from_graph)

    def kwargs_from_graph(self, graph):
        return {
            "fun": self.wrapper(numericalize(graph, 'loss'), **self.wrapper_kwargs),
            "jac": self.wrapper(numericalize(graph, 'loss', d_order=1), **self.wrapper_kwargs),
            "x0" : numerical_parameters(graph),
        }


class CliminOptimizer(TheanoOptimizer):

    def __init__(self, wrapper=(lambda f, **wrapper_kwargs: f), **wrapper_kwargs):
        """
        Constructs a generic climin optimizer, defaulting to no wrapper

        For mini-batch version with averaging use e.g. ``wrapper=compose(batch, average)``
        (no online wrapper, as climin already supports online natively)

        For setting wrapper_kwargs dnymically, for instance do:
            >>> myadvancedoptimizer.wrapper_kwargs['repeat_n_times'] = 4

        Parameters
        ----------
        wrapper: higher level function
            wrapper to manipulate standard translations
        wrapper_kwargs: dict
            kwargs for the wrapper, e.g. for setting repeat_n_times
        """
        self.wrapper = wrapper
        self.wrapper_kwargs = wrapper_kwargs
        super(CliminOptimizer, self).__init__(climin.util.optimizer, self.kwargs_from_graph)

    def kwargs_from_graph(self, graph):
        return {
            "f"     : self.wrapper(numericalize(graph, 'loss'), **self.wrapper_kwargs),
            "fprime": self.wrapper(numericalize(graph, 'loss', d_order=1), **self.wrapper_kwargs),
            "wrt"   : numerical_parameters(graph),
        }



class ScipyAnnealingOptimizer(TheanoOptimizer):

    def __init__(self, wrapper=(lambda f1, f2, **wrapper_kwargs: f1+f2), **wrapper_kwargs):
        """
        Constructs a generic anneallable scipy optimizer, defaulting to simply adding 'loss_data' + 'loss_regularizer'

        For mini-batch version with averaging and annealing use
            >>> wrapper=compose(online, batch, average, annealing)

        Set wrapper_kwargs dnymically like:
            >>> myoptimizer.wrapper_kwargs['repeat_n_times'] = 4

        Parameters
        ----------
        wrapper: higher level function
            wrapper to manipulate standard translations
        wrapper_kwargs: dict
            kwargs for the wrapper, e.g. for setting repeat_n_times

        Note
        ----
        the current abstract implementation has the disadvantage that the numerical values are saved twice
        once in ``numericalize('loss_data')`` and once in numericalize('loss_regularizer')
        a direct solution would be to outsource the saving of parameters into a new wrapper and make a simpler
        alternative to ``numericalize``, e.g. '_numericalize' (?)
        """
        self.wrapper = wrapper
        self.wrapper_kwargs = wrapper_kwargs
        super(ScipyOptimizer, self).__init__(scipy.optimize.minimize, self.kwargs_from_graph)

    def kwargs_from_graph(self, graph):
        return {
            "fun": self.wrapper(
                numericalize(graph, 'loss_data'),
                numericalize(graph, 'loss_regularizer'),
                **self.wrapper_kwargs
            ),
            "jac": self.wrapper(
                numericalize(graph, 'loss_data', d_order=1),
                numericalize(graph, 'loss_regularizer', d_order=1),
                **self.wrapper_kwargs
            ),
            "x0": numerical_parameters(graph),
        }


class CliminAnnealingOptimizer(TheanoOptimizer):

    def __init__(self, wrapper=(lambda f1, f2, **wrapper_kwargs: f1 + f2), **wrapper_kwargs):
        """
        Constructs a generic anneallable climin optimizer, defaulting to simply adding 'loss_data' + 'loss_regularizer'

        For mini-batch version with averaging and annealing use
            >>> wrapper=compose(batch, average, annealing)

        Set wrapper_kwargs dnymically like:
            >>> myoptimizer.wrapper_kwargs['repeat_n_times'] = 4

        Parameters
        ----------
        wrapper: higher level function
            wrapper to manipulate standard translations
        wrapper_kwargs: dict
            kwargs for the wrapper, e.g. for setting repeat_n_times

        Note
        ----http://mathworld.wolfram.com/images/equations/DiscreteDistribution/NumberedEquation1.gif
        the current abstract implementation has the disadvantage that the numerical values are saved twice
        once in ``numericalize('loss_data')`` and once in numericalize('loss_regularizer')
        a direct solution would be to outsource the saving of parameters into a new wrapper and make a simpler
        alternative to ``numericalize``, e.g. '_numericalize' (?)
        """
        self.wrapper = wrapper
        self.wrapper_kwargs = wrapper_kwargs
        super(CliminOptimizer, self).__init__(climin.util.optimizer, self.kwargs_from_graph)

    def kwargs_from_graph(self, graph):
        return {
            "f"     : self.wrapper(
                numericalize(graph, 'loss_data'),
                numericalize(graph, 'loss_regularizer'),
                **self.wrapper_kwargs
            ),
            "fprime": self.wrapper(
                numericalize(graph, 'loss_data', d_order=1),
                numericalize(graph, 'loss_regularizer', d_order=1),
                **self.wrapper_kwargs
            ),
            "wrt"   : numerical_parameters(graph),
        }


"""
Further Premaps
===============

Adding a regularizer is a common procedure to improve generalizability of the model (which is usually what we want).
The following defines a convenience postmap to easily adapt a given Model.
"""


def regularizer_L2(parameters):
    return sum((p**2).sum() for p in parameters)


def regularizer_L1(parameters):
    return sum(abs(p).sum() for p in parameters)


def regularizing_postmap(model, regularizer=regularizer_L2):
    """ postmap for a standard deterministic model

    Parameters
    ----------
    model : Model
        kwargs to be adapted by this postmap
    regularizer : function working on list of parameters, returning scalar loss
        shall regularize parameters. Alternatively you can specify some string identifiers for standard regularizers

    Returns
    -------
    IdentityDict
    """
    if isinstance(regularizer, basestring):
        try:
            regularizer = globals()["regularizer_%s" % regularizer]
        except KeyError:
            raise ValueError("unsupported regularizer string %s" % regularizer)

    return IdentityDict(
        lambda key: model[key],
        loss_data=model['loss'],
        loss_regularizer=regularizer(model['parameters']),
        loss=model['loss'] + regularizer(model['parameters'])
    )