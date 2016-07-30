#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import sys
import operator as op
import itertools
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
from schlichtanders.mycontextmanagers import ignored
from schlichtanders.myfunctools import convert
from schlichtanders.mylists import remove_duplicates, shallowflatten, add_up, remove, getall
from schlichtanders.mymeta import proxify, Proxifier
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

from schlichtanders.mylists import remove_duplicates

from theano.tensor.elemwise import Elemwise
from theano import scalar
__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
major proxifying
================
"""

def is_theano_proxified(o):
    return isinstance(o, Proxifier) or (hasattr(o, 'proxified') and o.proxified)

theano.tensor.opt.local_add_mul_fusion

def theano_proxify(o, n, weak_identity=True, reproxify="ignore"):
    """ wrapper arround proxify to handle theano specifities, especially type consistency

    Parameters
    ----------
    o : theano variable
        to be proxified
    n : theano variable
        target
    weak_identity : bool
        indicating whether additional clear manipulations should be applied to make n having the same type as o
        (e.g. adapting broadcasting, which is usually wanted)
    reproxify : str
        indicating what shell be done when a proxified ``o`` is be proxified again
        possible values ("warn", "raise", "print", "ignore")

    Returns
    -------
    proxified ``o`` in analogy to ``proxify``

    """
    if is_theano_proxified(o):
        if reproxify == "raise":
            raise RuntimeError(
                "(%s) is already proxified. It is sometimes not intended to proxify things twice." % o)
        elif reproxify == "warn":
            warnings.warn("(%s) is already proxified. It is sometimes not intended to proxify things twice." % o)
        elif reproxify == "print":
            print("(%s) is already proxified. It is sometimes not intended to proxify things twice." % o)
        #else do nothing

    def len_min_sum(iterable):
        return len(iterable) - sum(iterable)

    n.proxified = True  # only needed for week identity, but may be useful for both
    if weak_identity:
        name = n.name
        if (o.broadcastable != n.broadcastable
                and len_min_sum(o.broadcastable) == len_min_sum(n.broadcastable)):  # counts False
            idx = itertools.count()

            def broadcast_pattern():
                for b in o.broadcastable:
                    if b:
                        yield 'x'
                    else:
                        yield next(idx)

            n = n.squeeze().dimshuffle(*broadcast_pattern())
        if o.broadcastable != n.broadcastable:
            n = T.addbroadcast(n, *broadcastable_to_idx(o.broadcastable))
            n = T.unbroadcast(n, *unbroadcastable_to_idx(o.broadcastable))
        n.name = name

    assert o.type == n.type, "No proxifying as theano types differ. %s != %s" % (o.type, n.type)
    return proxify(o, n)

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
    if name is not None:  # some internal problems that this is not always the case?
        ret.name = name
    # this is decisive as our "constants" should be replacable usually,
    # however a constant won't get cloned
    # the least effort is to add an extra trivial layer which makes a constant a normal variable
    # this layer gets optimized away when compiling the function
    if isinstance(ret, TensorConstant):
        quasi_constant = T.tensor_copy(ret)  # = elemwise identity
        quasi_constant.name = ret.name if ret.name is not None else str(ret)
        return quasi_constant
    return ret


def is_pseudo_constant(var):
    return (var.owner is not None
            and var.owner.op == theano.tensor.tensor_copy
            and isinstance(var.owner.inputs[0], gof.Constant))


def broadcastable_to_idx(br):
    return [i for i, b in izip(itertools.count(), br) if b]

def unbroadcastable_to_idx(br):
    return [i for i, b in izip(itertools.count(), br) if not b]

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


"""
clone with replacing random variables
"""


def is_random_variable(v):
    return hasattr(v, 'rng')

def list_random_sources(outputs):
    return list(gen_variables(outputs, yield_on=lambda v: is_random_variable(v)))


@wraps(clone)
def clone_renew_rng(output,
          replace=None,
          strict=True,
          share_inputs=True,
          copy_inputs=DEPRECATED_ARG):
    replace_rv = {}
    for a in gen_variables(output, yield_on=lambda v: hasattr(v, 'rng')):
        rng1 = copy(a.rng)
        rng2 = copy(rng1)
        b = clone(a, replace={a.owner.inputs[0]: rng1})
        b.rng = rng1
        b.update = (rng1, rng2)
        replace_rv[a] = b
    if replace:
        replace_rv.update(replace)
    return clone(output, replace=replace_rv, strict=strict, share_inputs=share_inputs, copy_inputs=copy_inputs)


"""
monkey patch OpFromGraph to support outer input arguments
"""

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
    if isinstance(initial_variables, Sequence):
        agenda = list(initial_variables)
    else:
        agenda = [initial_variables]
    # uniques stores all unique values found overall, while agenda only stores those which are still to come
    remove_duplicates(agenda)
    uniques = set(agenda)
    while agenda:
        v = agenda.pop(0)
        if v.owner is None:  # don't yield None values
            continue
        if yield_on(v.owner):
            yield v.owner
        if stop_on(v.owner):
            continue

        for i in v.owner.inputs:
            if i not in uniques:
                agenda.append(i)
                uniques.add(i)


# # depth first search:
# def gen_nodes(initial_variables, yield_on=lambda n: True, stop_on=lambda v: False):
#     initial_variables = convert(initial_variables, Sequence)
#     for v in initial_variables:
#         for _v in _gen_nodes(v, yield_on=yield_on, stop_on=stop_on):  # yield from
#             yield _v
#
#
# def _gen_nodes(v, yield_on=lambda n: True, stop_on=lambda v: False):
#     if v.owner is None:
#         return
#     if yield_on(v.owner):
#         yield v.owner
#     if stop_on(v.owner):
#         return
#     for _v in v.owner.inputs:
#         for n in _gen_nodes(_v, yield_on=yield_on, stop_on=stop_on):  # yield from
#             yield n


def gen_variables(initial_variables, yield_on=lambda v: True, stop_on=lambda v: False, include_initials=True):
    if isinstance(initial_variables, Sequence):
        agenda = list(initial_variables)
    else:
        agenda = [initial_variables]
    if not include_initials:
        # make sure there are no duplicates
        agenda = add_up(v.owner.inputs for v in agenda if v.owner is not None)
    remove_duplicates(agenda)
    uniques = set(agenda)
    while agenda:
        v = agenda.pop(0)
        if yield_on(v):
            yield v
        if stop_on(v) or v.owner is None:
            continue

        for i in v.owner.inputs:
            if i not in uniques:
                agenda.append(i)
                uniques.add(i)


# # depth first search:
# def gen_variables(initial_variables, yield_on=lambda v: True, stop_on=lambda v: False, include_initials=True):
#     """ first level is not tested """
#     initial_variables = convert(initial_variables, Sequence)
#     if include_initials:
#         for v in initial_variables:
#             for _v in _gen_variables(v, yield_on=yield_on, stop_on=stop_on):  # yield from
#                 yield _v
#     else:
#         for v in initial_variables:
#             if v.owner is not None:
#                 for _v in v.owner.inputs:
#                     for __v in _gen_variables(_v, yield_on=yield_on, stop_on=stop_on):  #yield from
#                         yield __v
#
#
# def _gen_variables(v, yield_on=lambda v: v.owner is None, stop_on=lambda v: False):
#     if yield_on(v):
#         yield v
#     if v.owner is None:
#         return
#     if stop_on(v):
#         return
#     for _v in v.owner.inputs:
#         for __v in _gen_variables(_v, yield_on=yield_on, stop_on=stop_on):  #yield from
#             yield __v


def get_graph_inputs(variables):
    def leaf(v):
        return v.owner is None or is_pseudo_constant(v)
    return list(gen_variables(variables, yield_on=leaf, stop_on=leaf))


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


"""
intersecting graphs
===================
"""

def independent_subgraphs(inputs1, inputs2, outputs):
    """
    computes subgraphs for inputs1 only and inputs2 only
    Parameters
    ----------
    inputs1 : list of theano variables
    inputs2 : list of theano variables
    outputs : list of theano variables or single theano variable

    Returns
    -------
    tuple with entry being list of theano-expressions which denote the end of the respective inputsX-only subgraph
    """

    # TODO try to apply Canonicalize Add/Prod optimizer first and than extract subparts of Add/Prod
    # the reason is that if there is a node Add(....) then the current algorithm will only extract the inputs as
    # part of the independent_subgraph. However it would be far more efficient to split the Add in those which are
    # precomputable and those which are not

    outputs = convert(outputs, Sequence)
    if not hasattr(outputs[0], 'clients'):
        print("SETTING CLIENTS")
        for n in gen_nodes(outputs):
            for i, inp in enumerate(n.inputs):
                if not hasattr(inp, 'clients'):
                    inp.clients = []
                inp.clients.append((n,i))
        for i, out in enumerate(outputs):
            out.client = "output", i

    # reversed descendants as we will search from outputs backwards:
    descendants1 = _collect_descendants(inputs1)[::-1]
    descendants2 = _collect_descendants(inputs2)[::-1]

    independent_subgraphs1 = []
    independent_subgraphs2 = []

    # collect these subgraphs:
    # ------------------------
    agenda = list(outputs)
    remove_duplicates(agenda)
    uniques = set(agenda)  # pass each variable (at most) once
    while agenda:
        o = agenda.pop(0)
        in1, in2 = True, True
        try:
            descendants1.remove(o)
        except ValueError:
            in1 = False
        try:
            descendants2.remove(o)
        except ValueError:
            in2 = False

        if in1 and not in2:
            independent_subgraphs1.append(o)
        elif not in1 and in2:
            independent_subgraphs2.append(o)
        elif in1 and in2 and o.owner is not None:
            for i in o.owner.inputs:
                if i not in uniques:
                    agenda.append(i)
                    uniques.add(i)
        # else  not in1 and not in2 -> nothing to do

    return independent_subgraphs1, independent_subgraphs2


def _collect_descendants(inputs):
    descendants = list(inputs)
    remove_duplicates(descendants)
    uniques = set(descendants)
    for d in descendants:
        try:
            for n,i in d.clients:
                for o in n.outputs:
                    if o not in uniques:
                        descendants.append(o)
                        uniques.add(o)
        except AttributeError:
            pass
    return descendants  # unique and width-first sorted variables


def contains_node(top_expressions, sub_expression):
    try:
        next(gen_nodes(top_expressions, yield_on=lambda v:v==sub_expression))
        return True
    except StopIteration:
        return False

def contains_var(top_expressions, sub_expression):
    try:
        next(gen_variables(top_expressions, yield_on=lambda v:v==sub_expression))
        return True
    except StopIteration:
        return False

def independent_subgraphs_extend_add_mul(sub):
    """ extends a subgraph (list of theano variables, labeled with `_clients`
    towards their use in subsequent add/mul operators

    This is meant to be combined with local_add_mul_fusion"""
    # TODO include constants within add/mul extension
    if not hasattr(sub[0], 'clients'):
        raise ValueError("need client information. E.g. run gof.FunctionGraph before")
    
    all_client_nodes = reduce(op.add, ([c[0] for c in s.clients] for s in sub))
    all_client_nodes = remove_duplicates(all_client_nodes)
    # further remove all clients which part of another sub (this may happen as a free variable can also be used within a more complex sub)
    remove(all_client_nodes, key=lambda c: contains_node(sub, c))

    new_sub = []
    for node in all_client_nodes:
        if isinstance(node.op, Elemwise) and isinstance(node.op.scalar_op, (scalar.Add, scalar.Mul)):
            sub_i = []
            other_i = []
            for i in node.inputs:
                if i in sub:
                    sub_i.append(i)
                else:
                    other_i.append(i)

            assert len(other_i) >= 1, "if not, this node should have been part of sub before"
            if len(sub_i) >= 2:  # only if there is something to precompute, do the subgraph splitting
                s = node.op(*sub_i)
                new_sub.append(s)
                other_i.append(s)  # add it to other, so that it gets included
                new = node.op(*other_i)
                # update _clients
                for inp in sub_i + other_i[:-1]:
                    remove(inp.clients, key=lambda c: c[0] == node)
                for i, inp in enumerate(sub_i):
                    inp.clients.append((s.owner, i))
                for i, inp in enumerate(other_i[:-1]):  # don't include the last one, as this our new sub graph
                    inp.clients.append((new.owner, i))

                s.clients = [(new.owner, len(other_i)-1)]
                theano_proxify(node.outputs[0], new)  # add has only one output
            else:
                new_sub += sub_i
        else:
            new_sub += [i for i in node.inputs if i in sub]
    # remove duplicates
    new_sub = remove_duplicates(new_sub)

    # remove nested nodes
    i = 0
    while i < len(new_sub):
        s = new_sub[i]
        rest = new_sub[:i] + new_sub[i+1:]
        everywhere_nested = False
        if rest:
            everywhere_nested = all(contains_node(r, n) for n, _i in s.clients for r in rest)
        if everywhere_nested:
            del new_sub[i]
        else:
            i += 1
    return new_sub


def graphopt_merge_add_mul(inputs, outputs):
    mode = theano.compile.get_mode(None)
    everything_before_fusion = theano.compile.optdb.query(mode._optimizer, position_cutoff=49)
    # opt_merge = theano.gof.MergeOptimizer()  # theano.compile.optdb['merge1']
    # opt_canonicalize = theano.compile.optdb['canonicalize'].query(mode._optimizer)
    opt_add_mul_fusion = theano.tensor.opt.FusionOptimizer(theano.tensor.opt.local_add_mul_fusion)

    # stabilize = theano.compile.optdb['stabilize'].query(mode._optimizer)
    # print theano.compile.optdb['elemwise_fusion'].__position__
    # elemwise_fusion = theano.compile.optdb['elemwise_fusion'].query(mode._optimizer, position_cutoff=0.5)

    fg = theano.gof.FunctionGraph(inputs, outputs)
    # opt_merge(fg)
    # opt_canonicalize(fg)
    # opt_merge(fg)
    everything_before_fusion(fg)
    opt_add_mul_fusion(fg)
    # no further elemwise opt fusion
    return fg.inputs, fg.outputs


def rebuild_graph(inputs, outputs, *func_rebuilding):
    """ rebuild a graph represented by inputs, outputs, taking care of shared variables and random variables

    func_rebuilding work on FunctionGraph representation

    Parameters
    ----------
    all_inputs : list of theano.function inputs
        these are subgroups of inputs which should be preserved
    all_outputs : list of theano.function inputs
    func_rebuilding : homomorphisms func(inputs, outputs) -> inputs, outputs
        are applied in order
        can work on function graphs, i.e. don't have to handle shared variables or random variables

    Returns
    -------
    all_inputs, all_outputs, givens
    remapped versions plus givens, which need to be used when creating functions from the remappings
    (concerning shared variables and random variables)
    """
    Ni = len(inputs)
    placeholder_i = []
    placeholder_vars = []
    placeholder_orig = []
    for i, v in enumerate(inputs):
        if isinstance(v, SharedVariable) or is_random_variable(v):
            _v = v.type()
            placeholder_i.append(i)
            placeholder_vars.append(_v)
            placeholder_orig.append(v)
            inputs[i] = _v

    # add all not listed shared variables to the replacements (i.e. also not nested in the until now)
    extra_shared_orig = [v for v in gof.graph.inputs(outputs)
                         if isinstance(v, SharedVariable) and not contains_var(placeholder_orig, v)]
    extra_shared_vars = [v.type() for v in extra_shared_orig]
    placeholder_orig += extra_shared_orig
    placeholder_vars += extra_shared_vars
    placeholder_i += range(Ni, Ni+len(extra_shared_vars))
    inputs += extra_shared_vars  # add them to inputs so that if they are changed, we can track the change

    replace_dict = dict(izip(placeholder_orig, placeholder_vars))

    for i, v in enumerate(outputs):
        with ignored(KeyError):
            outputs[i] = replace_dict[v]

    inputs, outputs, extras = rebuild_collect_shared(outputs, inputs=inputs, replace=replace_dict, copy_inputs_over=False)
    # pre-optimize the graph, merging add and mul operators
    for f in func_rebuilding:
        inputs, outputs = f(inputs, outputs)

    placeholder_vars = getall(inputs, placeholder_i)  # might have changed concrete references
    reverse_dict = dict(izip(placeholder_vars, placeholder_orig))
    # TODO there are problems when using givens in theano.function. Concretely are random variables updated despite they
    # are not even included in the subgraph (some updating rules seem to get falsely copied)
    return inputs[:Ni], outputs, reverse_dict  # still best
    # TODO there are also problems when using theano.clone(..., replace=...) this doesn't preserve references of random variables... weird
    # outputs = theano.clone(outputs, replace=reverse_dict)
    # inputs = [reverse_dict[i] if i in reverse_dict else i for i in inputs[:Ni]]
    # return inputs, outputs


"""
Profiling
=========
"""

def get_profile(fct):
    if isinstance(fct, Function):
        mode = fct.maker.mode
        if not isinstance(mode, ProfileMode) or fct not in mode.profile_stats:
            mode = None
        if mode:
            return mode.profile_stats[fct]
        return getattr(fct, "profile", None)
    return None


