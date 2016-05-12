#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from six import integer_types
from collections import Sequence
from itertools import izip

import theano.tensor as T
import theano
from theano import config
from theano.tensor.basic import TensorType, as_tensor_variable
from theano.gof import Variable, utils
from theano.tensor.sharedvar import TensorSharedVariable, tensor_constructor
import numpy
import numpy as np

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

"""
general helpers
===============
"""

"""
reparameterization helpers
--------------------------
"""


def softplus(x, module=T):
    return module.log(module.exp(x) + 1)


def softplus_inv(y, module=np):
    return module.log(module.exp(y) - 1)


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


"""
reshape helpers
---------------
"""

def total_size(variables):
    return sum(v.size for v in variables)


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

"""
theano graph helpers
--------------------
"""

def gen_nodes(initial_variables, filter=lambda n:True):
    for v in initial_variables:
        if filter(v.owner):
            yield v.owner
        if v.owner is not None:
            for n in gen_nodes(v.owner.inputs, filter=filter):  # yield from
                yield n


def gen_variables(initial_variables, filter=lambda v:v.owner is None):
    for v in initial_variables:
        if filter(v):
            yield v
        if v.owner is not None:
            for _v in gen_variables(v.owner.inputs, filter=filter): #yield from
                yield _v


"""
shared redefined
================

with some additional convenience wrappers
"""

def shared(value, name=None, strict=False, allow_downcast=None, **kwargs):
    """Return a SharedVariable Variable, initialized with a copy or
    reference of `value`.

    This function iterates over constructor functions to find a
    suitable SharedVariable subclass.  The suitable one is the first
    constructor that accept the given value.  See the documentation of
    :func:`shared_constructor` for the definition of a contructor
    function.

    This function is meant as a convenient default.  If you want to use a
    specific shared variable constructor, consider calling it directly.

    ``theano.shared`` is a shortcut to this function.

    .. attribute:: constructors

    A list of shared variable constructors that will be tried in reverse
    order.

    Notes
    -----
    By passing kwargs, you effectively limit the set of potential constructors
    to those that can accept those kwargs.

    Some shared variable have ``borrow`` as extra kwargs.
    `See <http://deeplearning.net/software/theano/tutorial/aliasing.\
    html#borrowing-when-creating-shared-variables>`_ for details.

    Some shared variable have ``broadcastable`` as extra kwargs. As shared
    variable shapes can change, all dimensions default to not being
    broadcastable, even if ``value`` has a shape of 1 along some dimension.
    This parameter allows you to create for example a `row` or `column` 2d
    tensor.

    """

    try:
        for ctor in reversed(shared.constructors):
            try:
                var = ctor(value, name=name, strict=strict,
                           allow_downcast=allow_downcast, **kwargs)
                utils.add_tag_trace(var)
                return var
            except TypeError:
                continue

            # This may happen when kwargs were supplied
            # if kwargs were given, the generic_constructor won't be callable.
            #
            # This was done on purpose, the rationale being that if kwargs
            # were supplied, the user didn't want them to be ignored.

    except MemoryError as e:
        e.args = e.args + ('you might consider'
                           ' using \'theano.shared(..., borrow=True)\'',)
        raise

    raise TypeError('No suitable SharedVariable constructor could be found.'
                    ' Are you sure all kwargs are supported?'
                    ' We do not support the parameter dtype or type.'
                    ' value="%s". parameters="%s"' %
                    (value, kwargs))

shared.constructors = []


def shared_constructor(ctor, remove=False):
    if remove:
        shared.constructors.remove(ctor)
    else:
        if ctor not in shared.constructors:
            shared.constructors.append(ctor)
    return ctor


class ConstantShapeTensorSharedVariable(TensorSharedVariable):

    def __init__(self, name, type, value, strict,
                 allow_downcast=None, container=None, constant_shape=None):
        self._shape = constant_shape
        super(ConstantShapeTensorSharedVariable, self).__init__(
            name, type, value, strict, allow_downcast, container
        )

    @property
    def shape(self):  # overwrites _tensor_py_operators shape
        return self._shape

    # size refers by default to shape


@shared_constructor
def scalartensor_constructor(value, name=None, strict=False, allow_downcast=None,
                             borrow=False, broadcastable=None, target='cpu'):
    """
    SharedVariable Constructor for TensorType.

    Notes
    -----
    Regarding the inference of the broadcastable pattern...
    The default is to assume that the value might be resized in any
    dimension, so the default broadcastable is ``(False,)*len(value.shape)``.
    The optional `broadcastable` argument will override this default.

    """
    if target != 'cpu':
        raise TypeError('not for cpu')

    def check_type(value):
        return isinstance(value, (numpy.ndarray, numpy.number, float, integer_types, complex))
    if not check_type(value) and (isinstance(value, Sequence) and len(value) != 0
                                  and not isinstance(value[0], Sequence) and not check_type(value[0])):
        raise TypeError()

    value = numpy.array(value, copy=(not borrow))

    # if no broadcastable is given, then the default is to assume that
    # the value might be resized in any dimension in the future.
    #
    if broadcastable is None:
        broadcastable = (False,) * len(value.shape)
    type = TensorType(config.floatX, broadcastable=broadcastable)
    value = value.astype(config.floatX)
    # constant_shape = as_tensor_variable(value.shape)
    # TODO this gives a TheanoConstant, however the internal code requires either TheanoVariable or Tuple or List - Bug?
    # hence, use value.shape directly for now
    return ConstantShapeTensorSharedVariable(
        type=type,
        value=value,
        name=name,
        strict=strict,
        allow_downcast=allow_downcast,
        constant_shape=value.shape,
    )


class SymbolicSharedVariable(TensorSharedVariable):
    """ supports shape information """
    UNEVALUATED = []

    def __init__(self, name, type, value, strict,
                 allow_downcast=None, container=None, symbol=None, symbol_shape=None):
        SymbolicSharedVariable.UNEVALUATED.append(self)
        self.symbol = symbol
        self._shape = symbol_shape
        super(SymbolicSharedVariable, self).__init__(
            name, type, value, strict, allow_downcast, container
        )

    @property
    def shape(self):  # overwrites _tensor_py_operators shape
        return self._shape

    # size refers by default to shape

    def eval(self, inputs_to_values=None):
        if inputs_to_values is None:
            return super(SymbolicSharedVariable, self).eval()
        else:
            for sub_symbolic_shared_variable in gen_variables([self.symbol], filter=lambda v: isinstance(v, SymbolicSharedVariable)):
                sub_symbolic_shared_variable.eval(inputs_to_values)  # this goes recursively until the end
            value = self.symbol.eval(inputs_to_values).astype(config.floatX)
            self.set_value(value)

            try:
                SymbolicSharedVariable.UNEVALUATED.remove(self)
            except ValueError:
                pass  # not in list, which is fine
            return value

    @classmethod
    def evaluate_all_unevaluated(cls, inputs_to_values):
        while cls.UNEVALUATED:
            cls.UNEVALUATED[0].eval(inputs_to_values)


@shared_constructor
def symbol_constructor(value, name=None, strict=False, allow_downcast=None,
                       borrow=False, broadcastable=None, target='cpu'):

    """
    SharedVariable Constructor for TensorType.

    Notes
    -----
    Regarding the inference of the broadcastable pattern...
    The default is to assume that the value might be resized in any
    dimension, so the default broadcastable is ``(False,)*len(value.shape)``.
    The optional `broadcastable` argument will override this default.

    """
    if target != 'cpu':
        raise TypeError('not for cpu')

    if not isinstance(value, theano.gof.Variable):
        raise TypeError()
    symbol = value

    # if no broadcastable is given, then the default is to assume that
    # the value might be resized in any dimension in the future.
    #
    if broadcastable is None:
        broadcastable = value.type.broadcastable
    type = TensorType(config.floatX, broadcastable=broadcastable)  # use given broadcastable alternatively
    value = np.zeros((1,) * len(broadcastable), dtype=config.floatX)
    return SymbolicSharedVariable(
        type=type,
        value=numpy.array(value, copy=(not borrow)),
        name=name,
        strict=strict,
        allow_downcast=allow_downcast,
        symbol=symbol,
        symbol_shape=symbol.shape
    )


#: convenience alias as long as symbolic variables are prefiltered
sshared = symbol_constructor
