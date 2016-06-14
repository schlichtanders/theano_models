#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from six import integer_types
from collections import Sequence, OrderedDict, defaultdict
from itertools import izip
import inspect
from functools import wraps
from copy import deepcopy, copy
from time import time
import warnings

import numpy
import numpy as np

import theano
from theano import config, gof, clone as _clone
import theano.tensor as T
from theano.tensor.basic import as_tensor_variable as _as_tensor_variable
from theano.tensor.var import TensorConstant
from theano.tensor.sharedvar import TensorSharedVariable, tensor_constructor
from theano.gof import utils
from theano.gof.graph import Variable
from theano.scan_module.scan_utils import DEPRECATED_ARG
from theano.compile.sharedvalue import SharedVariable
from theano.compile.builders import OpFromGraph
from theano.compile.function_module import FunctionMaker, orig_function
from theano.compile import SharedVariable, rebuild_collect_shared, Function
from theano.compile.profilemode import ProfileMode


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
    if name is not None:  # this seems not be the case
        ret.name = name
    if isinstance(ret, TensorConstant):
        quasi_constant = T.tensor_copy(ret)
        quasi_constant.name = ret.name if ret.name is not None else str(ret)
        return quasi_constant  # tensor_copy is only a (bad) name for tensor_identity
    else:
        return ret


def is_pseudo_constant(var):
    return (var.owner is not None
            and var.owner.op == theano.tensor.tensor_copy
            and isinstance(var.owner.inputs[0], gof.Constant))


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
from schlichtanders.mymeta import proxify
n = as_tensor_variable(1)
m = clone(n)
o = clone(m)  # identical to copy(n) kind of
p = clone(n + m)  # makes weird results
print n.eval(), m.eval(), o.eval(), p.eval()
proxify(n, p)
print n, n.eval()
"""


# @wraps(_clone)
# def __clone(output,
#           replace=None,
#           strict=True,
#           share_inputs=True,
#           copy_inputs=DEPRECATED_ARG):
#     if replace is None:  # TODO test again, whether this is truly needed!
#         cp = copy(output)
#         if cp.owner is not None:  # CRUCIAL: reference in owner must mirrow self!!
#             # CRUCIAL: original owner must mirrow original self, hence copy also owner
#             cp_owner = copy(cp.owner)  # need new reference to adapt outputs
#             cp_owner.outputs = copy(cp.owner.outputs)  # inputs can stay the same
#             cp.owner.outputs[cp.index] = cp
#         return cp
#     else:
#         return _clone(output,
#           replace=replace,
#           strict=strict,
#           share_inputs=share_inputs,
#           copy_inputs=copy_inputs)

# my own implementation seems to have a deep-sitting bug
clone = _clone

""" monkey patch OpFromGraph to support outer input arguments """

def OpFromGraph__init__(self, inputs, outputs, **kwargs):
    if not isinstance(outputs, list):
        raise TypeError('outputs must be list', outputs)
    for i in inputs + outputs:
        if not isinstance(i, gof.Variable):
            raise TypeError(
                'inputs and outputs must be Variable instances', i)
    if 'updates' in kwargs or 'givens' in kwargs:
        raise TypeError('updates and givens are not allowed in kwargs')

    # To support correctly shared variables the inner fct should
    # not see them. Otherwise their is problem with the gradient.
    self.shared_inputs = [var for var in gof.graph.inputs(outputs)
                          if isinstance(var, SharedVariable)]
    shared_vars = [var.type() for var in self.shared_inputs]
    new = rebuild_collect_shared(outputs, inputs=inputs + shared_vars,
                                 replace=dict(izip(self.shared_inputs,
                                                   shared_vars)),
                                 copy_inputs_over=False)
    (new_inputs, new_outputs,
     [clone_d, update_d, update_expr, shared_inputs]) = new
    assert len(new_inputs) == len(inputs) + len(self.shared_inputs)
    assert len(new_outputs) == len(outputs)
    assert not update_d
    assert not update_expr
    assert not shared_inputs

    self.clone_d = clone_d
    self.new_inputs = new_inputs
    self.new_outputs = new_outputs
    self.inputs = inputs
    self.outputs = outputs
    self.on_unused_input = kwargs.pop("on_unused_input", 'warn')  # needs to be set for extra input arguments to work
    self.kwargs = kwargs
    self.input_types = [input.type for input in inputs]
    self.output_types = [output.type for output in outputs]


def OpFromGraph_make_thunk(self, node, storage_map, compute_map, no_recycling):
    clone_to_true = {}
    # frame hack (would be better to have a clone_d object directly or something like that):
    for f in inspect.stack():
        frame_locals = f[0].f_locals
        # print(frame_locals)
        # if 'clone_d' in frame_locals:
        #     print("clone_d", {"%s%i"%(k, hash(k)): hash(v) for k, v in frame_locals['clone_d'].iteritems()})
        #     # this shows that inputs are not cloned here!
        if 'self' in frame_locals and isinstance(frame_locals['self'], FunctionMaker):
            true_inputs = [i.variable for i in frame_locals['self'].inputs]
            cloned_inputs = frame_locals['self'].fgraph.inputs
            clone_to_true = dict(zip(cloned_inputs, true_inputs))
            # print("my clone mapping", {"%s%i" % (k, hash(k)): hash(v) for k, v in my_clone_mapping.iteritems()})
            # print()

    extra_inputs = []
    new_extra_inputs = []
    for k, v in compute_map.iteritems():
        if v[0]:  # v is singleton list with boolean whether the variable is already computed
            try:
                new = self.clone_d[clone_to_true[k]]
                if new not in self.new_inputs:
                    extra_inputs.append(k)
                    new_extra_inputs.append(new)
            except KeyError:
                pass

    node.inputs += extra_inputs  # node uses same variables as compute_map
    ret = super(OpFromGraph, self).make_thunk(node, storage_map,
                                              compute_map, no_recycling)
    if not hasattr(self, "fn"):
        self.fn = orig_function(self.new_inputs + new_extra_inputs,
                                self.new_outputs,
                                on_unused_input=self.on_unused_input,
                                **self.kwargs)
    return ret


OpFromGraph.__init__ = OpFromGraph__init__
OpFromGraph.make_thunk = OpFromGraph_make_thunk


"""
theano graph traverse helpers
-----------------------------
"""


def gen_nodes(initial_variables, yield_on=lambda n: True, stop_on=lambda v: False):
    if not isinstance(initial_variables, Sequence):
        initial_variables = [initial_variables]
    for v in initial_variables:
        if v.owner is not None:
            for _v in v.owner.inputs:
                for n in _gen_nodes(_v, yield_on=yield_on, stop_on=stop_on):  # yield from
                    yield n


def _gen_nodes(v, yield_on=lambda n: True, stop_on=lambda v: False):
    if yield_on(v.owner):
        yield v.owner
    if stop_on(v.owner):
        return
    for _v in v.owner.inputs:
        if v.owner is not None:
            for n in _gen_nodes(_v, yield_on=yield_on, stop_on=stop_on):  # yield from
                yield n


def gen_variables(initial_variables, yield_on=lambda v: v.owner is None, stop_on=lambda v: False):
    """ first level is not tested """
    if not isinstance(initial_variables, Sequence):
        initial_variables = [initial_variables]
    for v in initial_variables:
        if v.owner is not None:
            for _v in v.owner.inputs:
                for __v in _gen_variables(_v, yield_on=yield_on, stop_on=stop_on):  #yield from
                    yield __v

def _gen_variables(v, yield_on=lambda v: v.owner is None, stop_on=lambda v: False):
    if yield_on(v):
        yield v
    if stop_on(v):
        return
    if v.owner is not None:
        for _v in v.owner.inputs:
            for __v in _gen_variables(_v, yield_on=yield_on, stop_on=stop_on):  #yield from
                yield __v


# def depends_on(var1, var2):
#     for v in gen_variables(var1, lambda v: True):
#         if v == var2:
#             return True
#     return False
#
#
# def get_dependencies(variables, dependents=None):
#     if dependents is None:
#         dependents = variables
#     if not isinstance(variables, Sequence):
#         variables = [variables]
#     dependencies = defaultdict(list)  # {indepedent: dependent}
#     for var in variables:
#         for v in gen_variables(dependents, lambda v: v.owner is not None and var in v.owner.inputs):
#             dependencies[var].append(v)
#     return dependencies
#
#
# def sort_dependent_last(variables, return_idx=False, return_both=False):
#     """ sorts variables such that later variables depend on earlier (e.g. needed for flattening)
#     >>> a = as_tensor_variable(1)
#     >>> b = as_tensor_variable(2)
#     >>> c = b + 1
#     >>> d = c + b
#     >>> sort_dependent_last([c,a,b,d], return_idx=True)
#     [1, 2, 0, 3]
#
#     Parameters
#     ----------
#     variables : list of variables
#         to be sorted
#     return_idx : bool
#         of True, then a sorting index is returned instead of the sorted variables
#
#     Returns
#     -------
#     sorted idx if return_idx else sorted variables
#     """
#     variables = list(enumerate(variables))
#     sorted_v = []
#     sorted_i = []
#     while variables:
#         i, var = variables.pop(0)
#         if any(depends_on(var, v) for i, v in variables):  # initial var was popped
#             variables.append((i, var))  # put it to the back
#         else:
#             # do not depend on anything else
#             sorted_v.append(var)
#             sorted_i.append(i)
#     if return_idx:
#         return sorted_i
#     elif return_both:
#         return sorted_v, sorted_i
#     else:
#         return sorted_v


def get_inputs(variables):
    def leaf(v):
        return v.owner is None or is_pseudo_constant(v)
    return list(set(gen_variables(variables, yield_on=leaf, stop_on=leaf)))


"""
theano clone helpers
--------------------
"""

GroundedVariableType = (gof.graph.Constant, SharedVariable)
def is_clonable(variable):
    return variable.owner is not None or isinstance(variable, GroundedVariableType)


def clone_all(outputs):
    to_be_cloned = list(outputs)
    copies = {}
    while to_be_cloned:
        clone_recursive(to_be_cloned.pop(), to_be_cloned, copies, outputs)
    return [copies[o] for o in outputs]


def clone_recursive(o, to_be_cloned, copies, outputs):
    dependencies = {}
    for v in gen_variables(o, yield_on=lambda v: v in outputs, stop_on=lambda v: v in outputs):
        if v in to_be_cloned:
            to_be_cloned.remove(v)
            dependencies[v] = clone_recursive(v, to_be_cloned, copies, outputs)
        else:
            dependencies[v] = copies[v]
    o_cp = clone(o, replace=dependencies)
    o_cp.name = (o.name or str(o)) + "_copy"
    copies[o] = o_cp
    return o_cp
