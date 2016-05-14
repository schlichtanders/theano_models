#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from six import integer_types
from collections import Sequence, OrderedDict

import theano
from theano import config, gof, clone as _clone
import theano.tensor as T
from theano.tensor.basic import TensorType, as_tensor_variable
from theano.tensor.var import TensorConstant
from theano.gof import utils
from theano.tensor.sharedvar import TensorSharedVariable, tensor_constructor
from theano.compile.sharedvalue import SharedVariable
from theano.gof.graph import Variable
import numpy
import numpy as np
import warnings
from theano.tensor.basic import as_tensor_variable as _as_tensor_variable
from functools import wraps
from theano.scan_module.scan_utils import DEPRECATED_ARG
from copy import deepcopy, copy


__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
some bug fixes
--------------
"""

""" theano.tensor.var.TensorConstant cannot be used within theano.function[inputs, outputs]
this might make sense if this was called 'constant' but is an unneccesary distriction which is much better
to be handled by an convention instead.

Here we circument the current state by wrapping constants in an identity
"""


def as_tensor_variable(x, name=None, ndim=None):
    # for working with proxify it seems we need to ensure that everything has the same type
    # (there are internal checks like if c = a + b is initially int8 (because a and b were so)
    # and now a and b are proxified to mirrow float types, then c will break in ``assert c.dtype == (a+b).dtype``)
    if not isinstance(x, Variable):
        x = np.array(x, copy=False, dtype=config.floatX)
    ret = _as_tensor_variable(x, name, ndim)
    if isinstance(ret, TensorConstant):
        quasi_constant = T.tensor_copy(ret)
        quasi_constant.name = ret.name
        return quasi_constant  # tensor_copy is only a (bad) name for tensor_identity
    else:
        return ret


"""bugfixing __str__ of TensorConstant by monkey_patching"""

_old_str = TensorConstant.__str__
def _new_str(self):
    if self.name is not None:
        return self.name
    else:
        return _old_str(self)
TensorConstant.__str__ = _new_str


""" theano.clone gives unexpected bugs when using clone like copy in combination with proxify

from theano_models.util import clone
n = as_tensor_variable(1)
m = clone(n)
o = clone(m)  # identical to copy(n) kind of
p = clone(n + m)  # makes weird results
print n.eval(), m.eval(), o.eval(), p.eval()
proxify(n, p)
print n, n.eval()
"""

@wraps(_clone)
def clone(output,
          replace=None,
          strict=True,
          share_inputs=True,
          copy_inputs=DEPRECATED_ARG):
    if replace is None:
        cp = copy(output)
        if cp.owner is not None:  # CRUCIAL: reference in owner must mirrow self!!
            # CRUCIAL: original owner must mirrow original self, hence copy also owner
            cp_owner = copy(cp.owner)  # need new reference to adapt outputs
            cp_owner.outputs = copy(cp.owner.outputs)  # inputs can stay the same
            cp.owner.outputs[cp.index] = cp
        return cp
    else:
        return _clone(output,
          replace=replace,
          strict=strict,
          share_inputs=share_inputs,
          copy_inputs=copy_inputs)


"""
theano graph helpers
--------------------
"""

def gen_nodes(initial_variables, filter=lambda n:True):
    if not isinstance(initial_variables, Sequence):
        initial_variables = [initial_variables]
    for v in initial_variables:
        if filter(v.owner):
            yield v.owner
        if v.owner is not None:
            for n in gen_nodes(v.owner.inputs, filter=filter):  # yield from
                yield n


def gen_variables(initial_variables, filter=lambda v:v.owner is None):
    if not isinstance(initial_variables, Sequence):
        initial_variables = [initial_variables]
    for v in initial_variables:
        if filter(v):
            yield v
        if v.owner is not None:
            for _v in gen_variables(v.owner.inputs, filter=filter): #yield from
                yield _v


GroundedVariableType = (gof.graph.Constant, SharedVariable)


def is_clonable(variable):
    return variable.owner is not None or isinstance(variable, GroundedVariableType)


#
# """
# shared redefined
# ----------------
#
# with some additional convenience wrappers
# """
#
# def shared(value, name=None, strict=False, allow_downcast=None, **kwargs):
#     """Return a SharedVariable Variable, initialized with a copy or
#     reference of `value`.
#
#     This function iterates over constructor functions to find a
#     suitable SharedVariable subclass.  The suitable one is the first
#     constructor that accept the given value.  See the documentation of
#     :func:`shared_constructor` for the definition of a contructor
#     function.
#
#     This function is meant as a convenient default.  If you want to use a
#     specific shared variable constructor, consider calling it directly.
#
#     ``theano.shared`` is a shortcut to this function.
#
#     .. attribute:: constructors
#
#     A list of shared variable constructors that will be tried in reverse
#     order.
#
#     Notes
#     -----
#     By passing kwargs, you effectively limit the set of potential constructors
#     to those that can accept those kwargs.
#
#     Some shared variable have ``borrow`` as extra kwargs.
#     `See <http://deeplearning.net/software/theano/tutorial/aliasing.\
#     html#borrowing-when-creating-shared-variables>`_ for details.
#
#     Some shared variable have ``broadcastable`` as extra kwargs. As shared
#     variable shapes can change, all dimensions default to not being
#     broadcastable, even if ``value`` has a shape of 1 along some dimension.
#     This parameter allows you to create for example a `row` or `column` 2d
#     tensor.
#
#     """
#
#     try:
#         for ctor in reversed(shared.constructors):
#             try:
#                 var = ctor(value, name=name, strict=strict,
#                            allow_downcast=allow_downcast, **kwargs)
#                 utils.add_tag_trace(var)
#                 return var
#             except TypeError:
#                 continue
#
#             # This may happen when kwargs were supplied
#             # if kwargs were given, the generic_constructor won't be callable.
#             #
#             # This was done on purpose, the rationale being that if kwargs
#             # were supplied, the user didn't want them to be ignored.
#
#     except MemoryError as e:
#         e.args = e.args + ('you might consider'
#                            ' using \'theano.shared(..., borrow=True)\'',)
#         raise
#
#     raise TypeError('No suitable SharedVariable constructor could be found.'
#                     ' Are you sure all kwargs are supported?'
#                     ' We do not support the parameter dtype or type.'
#                     ' value="%s". parameters="%s"' %
#                     (value, kwargs))
#
# shared.constructors = []
#
#
# def shared_constructor(ctor, remove=False):
#     if remove:
#         shared.constructors.remove(ctor)
#     else:
#         if ctor not in shared.constructors:
#             shared.constructors.append(ctor)
#     return ctor
#
#
# class ConstantShapeTensorSharedVariable(TensorSharedVariable):
#
#     def __init__(self, name, type, value, strict,
#                  allow_downcast=None, container=None, constant_shape=None):
#         self._shape = constant_shape
#         super(ConstantShapeTensorSharedVariable, self).__init__(
#             name, type, value, strict, allow_downcast, container
#         )
#
#     @property
#     def shape(self):  # overwrites _tensor_py_operators shape
#         return self._shape
#
#     # size refers by default to shape
#
#
# @shared_constructor
# def scalartensor_constructor(value, name=None, strict=False, allow_downcast=None,
#                              borrow=False, broadcastable=None, target='cpu'):
#     """
#     SharedVariable Constructor for TensorType.
#
#     Notes
#     -----
#     Regarding the inference of the broadcastable pattern...
#     The default is to assume that the value might be resized in any
#     dimension, so the default broadcastable is ``(False,)*len(value.shape)``.
#     The optional `broadcastable` argument will override this default.
#
#     """
#     if target != 'cpu':
#         raise TypeError('not for cpu')
#
#     def check_type(value):
#         return isinstance(value, (numpy.ndarray, numpy.number, float, integer_types, complex))
#     if not check_type(value) and (isinstance(value, Sequence) and len(value) != 0
#                                   and not isinstance(value[0], Sequence) and not check_type(value[0])):
#         raise TypeError()
#
#     value = numpy.array(value, dtype=config.floatX, copy=(not borrow))
#
#     # if no broadcastable is given, then the default is to assume that
#     # the value might be resized in any dimension in the future.
#     #
#     if broadcastable is None:
#         broadcastable = (False,) * len(value.shape)
#     type = TensorType(config.floatX, broadcastable=broadcastable)
#     # constant_shape = as_tensor_variable(value.shape)
#     # TODO this gives a TheanoConstant, however the internal code requires either TheanoVariable or Tuple or List - Bug?
#     # hence, use value.shape directly for now
#     return ConstantShapeTensorSharedVariable(
#         type=type,
#         value=value,
#         name=name,
#         strict=strict,
#         allow_downcast=allow_downcast,
#         constant_shape=value.shape,
#     )
#
# '''
# class SymbolicSharedVariable(TensorSharedVariable):
#     """ supports shape information """
#     UNEVALUATED = []
#
#     def __init__(self, name, type, value, strict,
#                  allow_downcast=None, container=None, symbol=None, symbol_shape=None):
#         SymbolicSharedVariable.UNEVALUATED.append(self)
#         self.symbol = symbol
#         self._shape = symbol_shape
#         super(SymbolicSharedVariable, self).__init__(
#             name, type, value, strict, allow_downcast, container
#         )
#
#     @property
#     def shape(self):  # overwrites _tensor_py_operators shape
#         return self._shape
#
#     # size refers by default to shape
#
#     def eval(self, inputs_to_values=None):
#         if inputs_to_values is None:
#             return super(SymbolicSharedVariable, self).eval()
#         else:
#             for sub_symbolic_shared_variable in gen_variables([self.symbol], filter=lambda v: isinstance(v, SymbolicSharedVariable)):
#                 sub_symbolic_shared_variable.eval(inputs_to_values)  # this goes recursively until the end
#             value = self.symbol.eval(inputs_to_values).astype(config.floatX)
#             self.set_value(value)
#
#             try:
#                 SymbolicSharedVariable.UNEVALUATED.remove(self)
#             except ValueError:
#                 pass  # not in list, which is fine
#             return value
#
#     @classmethod
#     def evaluate_all_unevaluated(cls, inputs_to_values):
#         while cls.UNEVALUATED:
#             try:
#                 cls.UNEVALUATED[0].eval(inputs_to_values)
#             except AttributeError as e:
#                 warnings.warn("ignored AttributeError, as this seems to happen because of theanos internal copy system "
#                               "which re-generates variables again and again, however incomplete versions")
#                 del cls.UNEVALUATED[0]
#
#
# @shared_constructor
# def symbol_constructor(value, name=None, strict=False, allow_downcast=None,
#                        borrow=False, broadcastable=None, target='cpu'):
#
#     """
#     SharedVariable Constructor for TensorType.
#
#     Notes
#     -----
#     Regarding the inference of the broadcastable pattern...
#     The default is to assume that the value might be resized in any
#     dimension, so the default broadcastable is ``(False,)*len(value.shape)``.
#     The optional `broadcastable` argument will override this default.
#
#     """
#     if target != 'cpu':
#         raise TypeError('not for cpu')
#
#     if not isinstance(value, theano.gof.Variable):
#         raise TypeError()
#     symbol = value
#
#     # if no broadcastable is given, then the default is to assume that
#     # the value might be resized in any dimension in the future.
#     #
#     if broadcastable is None:
#         broadcastable = value.type.broadcastable
#     type = TensorType(config.floatX, broadcastable=broadcastable)  # use given broadcastable alternatively
#     value = np.zeros((1,) * len(broadcastable), dtype=config.floatX)
#     return SymbolicSharedVariable(
#         type=type,
#         value=numpy.array(value, copy=(not borrow)),
#         name=name,
#         strict=strict,
#         allow_downcast=allow_downcast,
#         symbol=symbol,
#         symbol_shape=symbol.shape
#     )
# '''
#
# symbolic_shared_variables = {}
#
# @shared_constructor
# def symbol_constructor2(value, name=None, strict=False, allow_downcast=None,
#                        borrow=False, broadcastable=None, target='cpu'):
#
#     """
#     SharedVariable Constructor for TensorType.
#
#     Notes
#     -----
#     Regarding the inference of the broadcastable pattern...
#     The default is to assume that the value might be resized in any
#     dimension, so the default broadcastable is ``(False,)*len(value.shape)``.
#     The optional `broadcastable` argument will override this default.
#
#     """
#     if target != 'cpu':
#         raise TypeError('not for cpu')
#
#     if not isinstance(value, theano.gof.Variable):
#         raise TypeError()
#     symbol = value
#
#     # if no broadcastable is given, then the default is to assume that
#     # the value might be resized in any dimension in the future.
#     #
#     if broadcastable is None:
#         broadcastable = value.type.broadcastable
#     type = TensorType(config.floatX, broadcastable=broadcastable)  # use given broadcastable alternatively
#     value = np.zeros((1,) * len(broadcastable), dtype=config.floatX)
#     new_shared_variable = ConstantShapeTensorSharedVariable(
#         type=type,
#         value=numpy.array(value, copy=(not borrow)),
#         name=name,
#         strict=strict,
#         allow_downcast=allow_downcast,
#         constant_shape=symbol.shape
#     )
#     symbolic_shared_variables[new_shared_variable] = symbol
#     return new_shared_variable
#
#
# def update_symbolic_var(sym_var, inputs_to_values=None):
#     if not isinstance(sym_var, TensorSharedVariable):  # reference changed type
#         del symbolic_shared_variables[sym_var]
#         return
#     symbol = symbolic_shared_variables[sym_var]
#     for sub_sym_var in gen_variables([symbol], filter=lambda v: v in symbolic_shared_variables):
#         update_symbolic_var(sub_sym_var, inputs_to_values)  # this goes recursively until the end
#     value = symbol.eval(inputs_to_values).astype(config.floatX)
#     sym_var.set_value(value, borrow=True)
#     return value
#
#
# def _sort_all_symbolic_var(sorted_keys, current_keys):
#     # first sort them, so that those without dependencies are first and later only depends on earlier
#     for sym_var in current_keys:
#         if sym_var in sorted_keys:
#             continue
#         elif not isinstance(sym_var, SharedVariable):  # reference changed type
#             del symbolic_shared_variables[sym_var]
#             continue
#         symbol = symbolic_shared_variables[sym_var]
#         sub_sym_vars = list(gen_variables([symbol], filter=lambda v: v in symbolic_shared_variables and v not in sorted_keys))
#         _sort_all_symbolic_var(sorted_keys, sub_sym_vars)  # this goes recursively until the end
#         sorted_keys.append(sym_var)
#
#
# def update_all_symbolic_var(inputs_to_values):
#     sorted_keys = []
#     _sort_all_symbolic_var(sorted_keys, symbolic_shared_variables.keys())
#
#     for sym_var in sorted_keys:
#         symbol = symbolic_shared_variables[sym_var]
#         value = symbol.eval(inputs_to_values).astype(config.floatX)
#         sym_var.set_value(value, borrow=True)
#