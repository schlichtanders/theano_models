#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from collections import Sequence

from subgraphs import subgraph
from itertools import izip
from util import as_tensor_variable, clone
import theano.tensor as T
from schlichtanders.myfunctools import convert
__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Concrete Helper Subgraphs
-------------------------
"""

eps = as_tensor_variable(0.0001)

@subgraph()
def softplus(x, module=T):
    return module.log(module.exp(x) + 1)

@subgraph()
def softplus_inv(y, module=T):
    return module.log(module.exp(y) - 1)

@subgraph()
def squareplus(x, module=T):
    return module.square(x) + eps  # to ensure >= 0

@subgraph()
def squareplus_inv(x, module=T):
    return module.sqrt(x - eps)


"""
norms and distances
-------------------
"""

@subgraph()
def L1(parameters):
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += abs(p).sum()
    return summed_up / n

@subgraph()
def L2(parameters):
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += (p**2).sum()
    return summed_up / n


def norm_distance(norm=L2):
    @subgraph()
    def distance(targets, outputs):
        """ targets and outputs are assumed to be *lists* of theano variables """
        return norm([t - o for t, o in izip(targets, outputs)])
    return distance


"""
reshape helpers
---------------
"""

@subgraph()
def total_size(variables):
    """ clones by default, as this function is usually used when something is meant to be replaced afterwards """
    variables = convert(variables, Sequence)
    return T.add(*(clone(v).size for v in variables))

@subgraph()
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
