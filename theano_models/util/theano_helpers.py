#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import sys
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
from theano import tensor

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
        for _v in _gen_nodes(v, yield_on=yield_on, stop_on=stop_on):  # yield from
            yield _v


def _gen_nodes(v, yield_on=lambda n: True, stop_on=lambda v: False):
    if yield_on(v.owner):
        yield v.owner
    if stop_on(v.owner):
        return
    for _v in v.owner.inputs:
        if v.owner is not None:
            for n in _gen_nodes(_v, yield_on=yield_on, stop_on=stop_on):  # yield from
                yield n


def gen_variables(initial_variables, yield_on=lambda v: v.owner is None, stop_on=lambda v: False, include_initials=True):
    """ first level is not tested """
    if not isinstance(initial_variables, Sequence):
        initial_variables = [initial_variables]
    if include_initials:
        for v in initial_variables:
            for _v in _gen_variables(v, yield_on=yield_on, stop_on=stop_on):  # yield from
                yield _v
    else:
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
    for v in gen_variables(o, yield_on=lambda v: v in outputs, stop_on=lambda v: v in outputs, include_initials=False):
        if v in to_be_cloned:
            to_be_cloned.remove(v)
            dependencies[v] = clone_recursive(v, to_be_cloned, copies, outputs)
        else:
            dependencies[v] = copies[v]
    o_cp = clone(o, replace=dependencies)
    o_cp.name = (o.name or str(o)) + "_copy"
    copies[o] = o_cp
    return o_cp



"""
Alternative Random Number Generator
-----------------------------------
Alternative to ``theano.tensor.shared_randomstreams.RandomStreams``
"""
import theano
from theano.tensor.shared_randomstreams import RandomStreams


class PooledRandomStreams(object):

    def __init__(self, pool_size=int(1e8), num_rng=np.random.RandomState(), sym_rng=RandomStreams()):
        self.pool_size = pool_size
        self.pools = {}
        self.num_rng = num_rng
        self.sym_rng = sym_rng

    def _sample(self, key, shape):
        if key not in self.pools:
            self.pools[key] = as_tensor_variable(getattr(self.num_rng, key)(size=self.pool_size))

        size = T.prod(shape)
        start_i = self.sym_rng.random_integers(size=tuple(), low=0, high=self.pool_size - size - 1)  # -1 as also high is inclusive
        return T.reshape(self.pools[key][start_i:start_i + size], shape)

    # def binomial(self, size=None, n=1, p=0.5, ndim=None, dtype='int64', prob=None):
    #     """
    #     Sample n times with probability of success p for each trial and
    #     return the number of successes.
    #
    #     If the size argument is ambiguous on the number of dimensions,
    #     ndim may be a plain integer to supplement the missing information.
    #
    #     """
    #     if prob is not None:
    #         p = prob
    #         print("DEPRECATION WARNING: the parameter prob to the binomal fct have been renamed"
    #               "to p to have the same name as numpy.",
    #               file=sys.stderr)
    #     raise NotImplemented


    def uniform(self, size=None, low=0.0, high=1.0, ndim=None, dtype=None):
        """
        Sample a tensor of given size whose element from a uniform
        distribution between low and high.

        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing information.
        """
        offset = high - low
        standard = self._sample('uniform', shape=size)
        if low != 0.0:
            standard -= low
        if offset != 1.0:
            standard *= high - low
        return standard


    def normal(self, size=None, avg=0.0, std=1.0, ndim=None, dtype=None):
        """
        Sample from a normal distribution centered on avg with
        the specified standard deviation (std).

        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing information.

        """
        standard = self._sample('normal', shape=size)
        if std != 1.0:
            standard *= std
        if avg != 0.0:
            standard += avg
        return standard


    # def random_integers(self, size=None, low=0, high=1, ndim=None,
    #                     dtype='int64'):
    #     """
    #     Sample a random integer between low and high, both inclusive.
    #
    #     If the size argument is ambiguous on the number of dimensions,
    #     ndim may be a plain integer to supplement the missing information.
    #
    #     """
    #     raise NotImplemented
    #
    #
    # def choice(self, size=None, a=2, replace=True, p=None, ndim=None,
    #            dtype='int64'):
    #     """
    #     Choose values from `a` with or without replacement.
    #
    #     `a` can be a 1-D array or a positive scalar.
    #     If `a` is a scalar, the samples are drawn from the range 0,...,a-1.
    #
    #     If the size argument is ambiguous on the number of dimensions,
    #     ndim may be a plain integer to supplement the missing information.
    #
    #     """
    #     raise NotImplemented
    #
    #
    # def poisson(self, size=None, lam=None, ndim=None, dtype='int64'):
    #     """
    #     Draw samples from a Poisson distribution.
    #
    #     The Poisson distribution is the limit of the Binomial distribution for
    #     large N.
    #
    #     If the size argument is ambiguous on the number of dimensions,
    #     ndim may be a plain integer to supplement the missing information.
    #
    #     """
    #     raise NotImplemented
    #
    #
    # def permutation(self, size=None, n=1, ndim=None, dtype='int64'):
    #     """
    #     Return permutations of the integers between 0 and n-1.
    #
    #     Returns them as many times as required by size. For instance,
    #     if size=(p,q), p*q permutations will be generated,
    #     and the output shape will be (p,q,n), because each
    #     permutation is of size n.
    #
    #     Theano tries to infer the number of dimensions from the length
    #     of the size argument and the shape of n, but you may always
    #     specify it with the `ndim` parameter.
    #
    #     Notes
    #     -----
    #     Note that the output will then be of dimension ndim+1.
    #
    #     """
    #     raise NotImplemented
    #
    #
    # def multinomial(self, size=None, n=1, pvals=[0.5, 0.5], ndim=None,
    #                 dtype='int64'):
    #     """
    #     Sample n times from a multinomial distribution defined by
    #     probabilities pvals, as many times as required by size. For
    #     instance, if size=(p,q), p*q samples will be drawn, and the
    #     output shape will be (p,q,len(pvals)).
    #
    #     Theano tries to infer the number of dimensions from the length
    #     of the size argument and the shapes of n and pvals, but you may
    #     always specify it with the `ndim` parameter.
    #
    #     Notes
    #     -----
    #     Note that the output will then be of dimension ndim+1.
    #
    #     """
    #     raise NotImplemented
    #
    #
    # def shuffle_row_elements(self, input):
    #     """
    #     Return a variable with every row (rightmost index) shuffled.
    #
    #     This uses permutation random variable internally, available via
    #     the ``.permutation`` attribute of the return value.
    #
    #     """
    #     perm = self.permutation(size=input.shape[:-1], n=input.shape[-1],
    #                             ndim=input.ndim - 1)
    #     shuffled = tensor.permute_row_elements(input, perm)
    #     shuffled.permutation = perm
    #     return shuffled
