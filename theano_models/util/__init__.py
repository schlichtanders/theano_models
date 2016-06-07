#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import izip
import theano.tensor as T
from theano_helpers import as_tensor_variable, clone, is_clonable, clone_all  # because this is so central
from collections import Sequence
from schlichtanders.mymeta import proxify
from schlichtanders.myfunctools import fmap
from schlichtanders.mylists import remove_duplicates
from schlichtanders.mydicts import update

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
reparameterization helpers
--------------------------
"""

eps = as_tensor_variable(0.0001)


def softplus(x, module=T):
    return module.log(module.exp(x) + 1)


def softplus_inv(y, module=T):
    return module.log(module.exp(y) - 1)


def squareplus(x, module=T):
    return module.square(x) + eps  # to ensure >= 0


def squareplus_inv(x, module=T):
    return module.sqrt(x - eps)


def reparameterize(parameters, f, finv):
    """
    use e.g. within a Merge
    >>> reparameterize(model['parameters_positive'], softplus, softplusinv)
    to get new parameters

    new_param = finv(param)
        param = f(new_param)

    Parameters
    ----------
    parameters : list of theano variables
        to be reparameterized
    f : function theano_variable -> theano_variable
    finv : function theano_variable -> theano_variable

    Returns
    -------
    new underlying parameters
    (i.e. NOT the reparameterized parameters, they are substituted, i.e. references still hold)
    """
    assert all(is_clonable(param) for param in parameters), (
        "Can only flatten clonable parameters."
    )
    new_underlying_parameters = []
    for param in parameters:
        cp_param = clone(param)
        cp_param.name = (cp_param.name or str(cp_param))  # + "_copy"
        new_param = finv(cp_param)  # clone is decisive as we otherwise get an infinite reference loop
        new_param.name = cp_param.name + "_" + f.func_name
        proxified_param = f(new_param)
        proxified_param.name = (param.name or str(param)) + "_reparam"
        proxify(param, proxified_param)
        new_underlying_parameters.append(new_param)
    return new_underlying_parameters


"""
norms and distances
-------------------
"""


def L1(parameters):
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += abs(p).sum()
    return summed_up / n


def L2(parameters):
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += (p**2).sum()
    return summed_up / n


def norm_distance(norm=L2):
    def distance(targets, outputs):
        """ targets and outputs are assumed to be *lists* of theano variables """
        return norm([t - o for t, o in izip(targets, outputs)])
    return distance


"""
reshape helpers
---------------
"""


def total_size(variables):
    """ clones by default, as this function is usually used when something is meant to be replaced afterwards """
    if not isinstance(variables, Sequence):
        variables = [variables]
    return sum(clone(v).size for v in variables)


def complex_reshape(vector, variables):
    """ reshapes vector into elements with shapes like variables

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
    # NOTE: this only works with ConstantShapeSharedVariables, as usually when ``variables`` get proxified, also the old shape refers to the new variables
    i = 0
    for v in variables:
        yield vector[i:i+v.size].reshape(v.shape)
        i += v.size

