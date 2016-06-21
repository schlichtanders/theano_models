#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from collections import Sequence

from subgraphs import subgraph
from itertools import izip
from util import as_tensor_variable, clone
import theano.tensor as T
from schlichtanders.myfunctools import convert
import numpy as np
__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Concrete Helper Subgraphs
-------------------------
"""
"""
for positive values:
"""
eps = as_tensor_variable(0.0001)

@subgraph
def softplus(x, module=T):
    return module.log(module.exp(x) + 1)

@subgraph
def softplus_inv(y, module=T):
    return module.log(module.exp(y) - 1)

@subgraph
def squareplus(x, module=T):
    return module.square(x) + eps  # to ensure >= 0

@subgraph
def squareplus_inv(x, module=T):
    return module.sqrt(x - eps)


"""
for p-values (0,1)
"""

@subgraph
def tan_01_R(x):
    return T.tan(np.pi * (x - 0.5))

@subgraph
def tan_01_R_inv(y):
    return T.ifelse(T.lt(y, 0), -(T.arctan(T.inv(y))) / np.pi, 1-(T.arctan(T.inv(y))) / np.pi)

@subgraph
def square_01_R(x):
    return (2*x - 1) / (x - x*x)

@subgraph
def square_01_R_inv(y, module=T):
    return (module.sqrt(y*y + 4) + y - 2) / (2*y)


@subgraph
def logit(x):
    return -T.log(T.inv(x) - 1)

@subgraph
def logistic(y):
    T.inv(1 + T.exp(-y))


"""
psumto1
"""
@subgraph
def softmax(y, module=T):
    expy = module.exp(y)
    return expy / expy.sum()


@subgraph
def softmax_inv(x, initial_normalization=1, module=T):
    return module.log(x*initial_normalization)


"""
norms and distances
-------------------
"""

@subgraph
def L1(parameters):
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += abs(p).sum()
    return summed_up / n

@subgraph
def L2(parameters):
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += (p**2).sum()
    return summed_up / n


def norm_distance(norm=L2):
    @subgraph
    def distance(targets, outputs):
        """ targets and outputs are assumed to be *lists* of theano variables """
        return norm([t - o for t, o in izip(targets, outputs)])
    return distance


"""
reshape helpers
---------------
"""

@subgraph
def total_size(variables):
    """ clones by default, as this function is usually used when something is meant to be replaced afterwards """
    variables = convert(variables, Sequence)
    return T.add(*(clone(v).size for v in variables))

@subgraph
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
    i = 0
    for v in variables:
        yield vector[i:i+v.size].reshape(v.shape)
        i += v.size
