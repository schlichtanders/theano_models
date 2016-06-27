#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import operator as op
import types
from collections import Sequence
from functools import wraps
from itertools import izip
import numpy as np
import wrapt

import theano
from schlichtanders.mycontextmanagers import until_stopped
from schlichtanders.mylists import remove_duplicates
from theano import gof
from theano.gof.fg import MissingInputError
import theano.tensor as T
from theano.gof.graph import Variable
from theano.compile import Function

from schlichtanders.myfunctools import convert
from base import Model, Merge, track_model, track_merge, get_inputting_references, get_outputting_references
from util import as_tensor_variable, clone, deepflatten_keep_vars

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Concrete Helper Models
----------------------
"""

"""
for positive values:
"""
eps = as_tensor_variable(1e-16)

@track_model
def softplus(x, module=T):
    return module.log(module.exp(x) + 1)

@track_model
def softplus_inv(y, module=T):
    return module.log(module.exp(y) - 1)

@track_model
def squareplus(x, module=T):
    return module.square(x) + eps  # to ensure >= 0

@track_model
def squareplus_inv(x, module=T):
    return module.sqrt(x - eps)


"""
for p-values (0,1)
"""

@track_model
def tan_01_R(x):
    return T.tan(np.pi * (x - 0.5))

@track_model
def tan_01_R_inv(y):
    return T.ifelse(T.lt(y, 0), -(T.arctan(T.inv(y))) / np.pi, 1-(T.arctan(T.inv(y))) / np.pi)

@track_model
def square_01_R(x):
    return (2*x - 1) / (x - x*x)

@track_model
def square_01_R_inv(y, module=T):
    return (module.sqrt(y*y + 4) + y - 2) / (2*y)


@track_model
def logit(x):
    return -T.log(T.inv(x) - 1)

@track_model
def logistic(y):
    T.inv(1 + T.exp(-y))


"""
psumto1
"""
@track_model
def softmax(y, module=T):
    expy = module.exp(y)
    return expy / expy.sum()


@track_model
def softmax_inv(x, initial_normalization=1, module=T):
    return module.log(x*initial_normalization)


"""
norms and distances
-------------------
"""

@track_model
def L1(parameters):
    parameters = convert(parameters, Sequence)
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += abs(p).sum()
    return summed_up / n

@track_model
def L2(parameters):
    parameters = convert(parameters, Sequence)
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += (p**2).sum()
    return summed_up / n


def norm_distance(norm=L2):
    @track_model
    def distance(targets, outputs):
        """ targets and outputs are assumed to be *lists* of theano variables """
        return norm([t - o for t, o in izip(targets, outputs)])
    return distance


"""
reshape helpers
---------------
"""

@track_model
def total_size(variables):
    """ clones by default, as this function is usually used when something is meant to be replaced afterwards """
    variables = convert(variables, Sequence)
    try:
        sizes = theano.function([], [v.size for v in variables], mode="FAST_COMPILE")()
    except MissingInputError:
        sizes = [clone(v).size for v in variables]
    return reduce(op.add, sizes)  # for generality to also include numerical sizes

@track_model
def complex_reshape(vector, variables):
    """ reshapes vector into elements with shapes like variables

    CAUTION: if you want to use this in combination with proxify, first clone the variables. Otherwise recursions occur.

    Parameters
    ----------
    vector : list
        shall be reshaped
    variables : list of theano variables
        .size and .shape will be used to reshape vector appropriately

    Returns
    -------
        reshaped parts of the vector
    """
    try:
        shapes = theano.function([], [v.shape for v in variables], mode="FAST_COMPILE")()
    except MissingInputError:
        shapes = [clone(v).shape for v in variables]

    i = 0
    for shape in shapes:
        size = np.prod(shape)
        yield vector[i:i+size].reshape(shape)
        i += size


"""
higher level helpers
--------------------
"""
def fct_to_inputs_outputs(th_graph):
    """ generic helper function to get inputs and outputs from a given ... """
    if isinstance(th_graph, Function):
        outputs = th_graph.maker.fgraph.outputs
        inputs = th_graph.maker.fgraph.inputs
    elif isinstance(th_graph, gof.FunctionGraph):
        outputs = th_graph.outputs
        inputs = th_graph.inputs
    elif isinstance(th_graph, tuple) and len(th_graph) == 2:
        inputs, outputs = th_graph
    elif isinstance(th_graph, list):  # assume each entry is a th_graph itself
        def subgraphs():
            for sub_graph in th_graph:
                inputs, outputs = fct_to_inputs_outputs(sub_graph)
                yield Model(ignore=True, inputs=inputs, outputs=outputs)
        inputs, outputs = fct_to_inputs_outputs(Merge(*subgraphs()))
    elif isinstance(th_graph, Model):
        # inputs, outputs = th_graph['inputs'], th_graph['outputs']
        inputs, outputs = get_inputting_references(th_graph), get_outputting_references(th_graph)
    else:
        if isinstance(th_graph, gof.Variable):
            th_graph = [th_graph]
        elif isinstance(th_graph, gof.Apply):
            th_graph = th_graph.outputs
        outputs = th_graph
        inputs = gof.graph.inputs(th_graph)

    if not isinstance(outputs, Sequence):
        outputs = [outputs]

    assert isinstance(inputs, (list, tuple))
    assert isinstance(outputs, (list, tuple))
    assert all(isinstance(v, gof.Variable) for v in inputs + outputs)
    return list(inputs), list(outputs)


def get_equiv_by_name(fct1, fct2):
    """ computes a mapping between to supported graph-formats by assuming unique names """
    inputs1, outputs1 = fct_to_inputs_outputs(fct1)
    inputs2, outputs2 = fct_to_inputs_outputs(fct2)

    fct1_to_name = {}
    for v1 in gof.graph.variables(inputs1, outputs1):
        if v1.name is not None:
            fct1_to_name[v1] = v1.name

    name_to_fct2 = {}
    for v2 in gof.graph.variables(inputs2, outputs2):
        if v2.name is not None:
            name_to_fct2[v2.name] = v2

    return {v1: name_to_fct2[n] for v1, n in fct1_to_name.iteritems()}




