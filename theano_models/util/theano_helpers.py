#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from six import integer_types
from collections import Sequence, OrderedDict

import theano
from theano import config
from theano.tensor.basic import TensorType, as_tensor_variable
from theano.gof import Variable, utils
from theano.tensor.sharedvar import TensorSharedVariable, tensor_constructor
from theano.compile.sharedvalue import SharedVariable
import numpy
import numpy as np
import warnings

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

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
----------------

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

    value = numpy.array(value, dtype=config.floatX, copy=(not borrow))

    # if no broadcastable is given, then the default is to assume that
    # the value might be resized in any dimension in the future.
    #
    if broadcastable is None:
        broadcastable = (False,) * len(value.shape)
    type = TensorType(config.floatX, broadcastable=broadcastable)
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

'''
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
            try:
                cls.UNEVALUATED[0].eval(inputs_to_values)
            except AttributeError as e:
                warnings.warn("ignored AttributeError, as this seems to happen because of theanos internal copy system "
                              "which re-generates variables again and again, however incomplete versions")
                del cls.UNEVALUATED[0]


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
'''

symbolic_shared_variables = {}

@shared_constructor
def symbol_constructor2(value, name=None, strict=False, allow_downcast=None,
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
    new_shared_variable = ConstantShapeTensorSharedVariable(
        type=type,
        value=numpy.array(value, copy=(not borrow)),
        name=name,
        strict=strict,
        allow_downcast=allow_downcast,
        constant_shape=symbol.shape
    )
    symbolic_shared_variables[new_shared_variable] = symbol
    return new_shared_variable


def update_symbolic_var(sym_var, inputs_to_values=None):
    if not isinstance(sym_var, TensorSharedVariable):  # reference changed type
        del symbolic_shared_variables[sym_var]
        return
    symbol = symbolic_shared_variables[sym_var]
    for sub_sym_var in gen_variables([symbol], filter=lambda v: v in symbolic_shared_variables):
        update_symbolic_var(sub_sym_var, inputs_to_values)  # this goes recursively until the end
    value = symbol.eval(inputs_to_values).astype(config.floatX)
    sym_var.set_value(value, borrow=True)
    return value


def _sort_all_symbolic_var(sorted_keys, current_keys):
    # first sort them, so that those without dependencies are first and later only depends on earlier
    for sym_var in current_keys:
        if sym_var in sorted_keys:
            continue
        elif not isinstance(sym_var, SharedVariable):  # reference changed type
            del symbolic_shared_variables[sym_var]
            continue
        symbol = symbolic_shared_variables[sym_var]
        sub_sym_vars = list(gen_variables([symbol], filter=lambda v: v in symbolic_shared_variables and v not in sorted_keys))
        _sort_all_symbolic_var(sorted_keys, sub_sym_vars)  # this goes recursively until the end
        sorted_keys.append(sym_var)


def update_all_symbolic_var(inputs_to_values):
    sorted_keys = []
    _sort_all_symbolic_var(sorted_keys, symbolic_shared_variables.keys())

    for sym_var in sorted_keys:
        symbol = symbolic_shared_variables[sym_var]
        value = symbol.eval(inputs_to_values).astype(config.floatX)
        sym_var.set_value(value, borrow=True)



# TODO rebuild theano_models such that no shared variables are used in first instance. replace ``shared`` with ``as_tensor_variable``