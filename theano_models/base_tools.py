#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import operator as op
import types
import warnings
from collections import Sequence
from functools import wraps
from itertools import izip
import numpy as np
import wrapt

import theano
from schlichtanders.mycontextmanagers import until_stopped
from schlichtanders.mylists import remove_duplicates
from schlichtanders.mymeta import Proxifier, proxify
from theano import gof
from theano.gof.fg import MissingInputError
import theano.tensor as T
from theano.gof.graph import Variable
from theano.compile import Function

from schlichtanders.myfunctools import convert
from base import Model, Merge, get_inputting_references, get_outputting_references, model_to_output
from theano_models.util.theano_helpers import broadcastable_to_idx, unbroadcastable_to_idx, theano_proxify, \
    is_theano_proxified
from util import as_tensor_variable, clone, deepflatten_keep_vars, is_clonable, clone_all

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Core Decorators
===============
To easily track functions as Models (e.g. for visualization).

By default no function is wrapped in this package, use the decorators manually as higher order functions.
"""


@wrapt.decorator
def as_model(wrapped, instance, args, kwargs):
    """ function wrapper which tracks each function execution as a Model

    with inputs, outputs set to the function's arguments / return values

    Parameters
    ----------
    name : str
        see Model, defaults to ``wrapped.func_name``
    everything else see Model

    Returns
    -------
    decorated function which returns the respective model.
    Combine it with ``model_to_output`` or use ``track_model`` directly to return the function output instead.
    """
    outputs = wrapped(*args, **kwargs)
    if isinstance(outputs, types.GeneratorType):
        outputs = list(outputs)

    return Model(
        name=wrapped.func_name,
        inputs=remove_duplicates(deepflatten_keep_vars(args)),
        outputs=outputs
    )

@wrapt.decorator
def track_model(wrapped, instance, args, kwargs):
    """ like ``as_model``, however the decorated functions returns its normal output instead of the respective model

    it is nevertheless tracked """
    outputs = wrapped(*args, **kwargs)
    if isinstance(outputs, types.GeneratorType):
        outputs = list(outputs)

    Model(
        name=wrapped.func_name,
        inputs=remove_duplicates(deepflatten_keep_vars(args)),
        outputs=outputs
    )
    return outputs


def as_merge(*merge_subgraphs, **merge_kwargs):
    """ Decorates a function to be listed as a Model
    Thereby the Model defined by the given Merge parameters is adapted by the function {inputs:..., outpupts:...}
    dictionary to create and list a new Model

    Parameters
    ----------
    track : bool
        see Merge, defaults to True
    name : str
        see Merge, defaults to concatination of first given ``subgraph.name`` with ``wrapped.func_name``
    everything else see Merge

    Returns
    -------
    decorated function which returns the respective Model instead of the output.
    Combine it with ``model_to_output`` or use ``track_merge`` directly to return the function output instead.
    """
    track = merge_kwargs.pop('track', True)
    def decorator(wrapped):
        name = merge_kwargs.pop('name', None)
        if name is None:
            try:
                name = next(sg.name + '.' for sg in merge_subgraphs if hasattr(sg, 'name'))
            except StopIteration:
                name = ""
            name += wrapped.func_name

        @wraps(wrapped)
        def wrapper(*args, **kwargs):
            outputs = wrapped(*args, **kwargs)
            if isinstance(outputs, types.GeneratorType):
                outputs = list(outputs)
            return Merge(*merge_subgraphs,
                  name=name, track=track,
                  inputs=list(args), outputs=outputs,
                  **merge_kwargs)
        return wrapper
    return decorator


def track_merge(*merge_subgraphs, **merge_kwargs):
    """ like ``as_merge``, however the decorated functions returns its normal output instead of the respective model

    it is nevertheless tracked """
    track = merge_kwargs.pop('track', True)
    def decorator(wrapped):
        name = merge_kwargs.pop('name', None)
        if name is None:
            try:
                name = next(sg.name + '.' for sg in merge_subgraphs if hasattr(sg, 'name'))
            except StopIteration:
                name = ""
            name += wrapped.func_name

        @wraps(wrapped)
        def wrapper(*args, **kwargs):
            outputs = wrapped(*args, **kwargs)
            if isinstance(outputs, types.GeneratorType):
                outputs = list(outputs)
            Merge(*merge_subgraphs,
                  name=name, track=track,
                  inputs=list(args), outputs=outputs,
                  **merge_kwargs)
            return outputs
        return wrapper
    return decorator


# proxifying function need reversed model

def track_proxmodel(key_underlying_params):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        refs = {
            'inputs': [],  # gets overwritten by key_underlying_params in case
            'outputs': args[0],
            key_underlying_params: wrapped(*args, **kwargs),
        }
        if isinstance(refs['outputs'], types.GeneratorType):
            refs['outputs'] = list(refs['outputs'])
        # track as model
        Model(name=wrapped.func_name,**refs)
        return refs['outputs']
    return wrapper


def as_proxmodel(key_underlying_params):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        refs = {
            'inputs': [],  # gets overwritten by key_underlying_params in case
            'outputs': args[0],
            key_underlying_params: wrapped(*args, **kwargs),
        }
        if isinstance(refs['outputs'], types.GeneratorType):
            refs['outputs'] = list(refs['outputs'])
        # track as model
        return Model(name=wrapped.func_name,**refs)
    return wrapper


"""
Proxifcation
============
proxifying functions which are used everywhere and always.
"""


def prox_reparameterize(parameters, f, finv, givens={}):
    """
    In General what is done is that for each param in parameters::
        new_param = finv(param)
            param = f(new_param)

    The underlying new_params are returned

    Parameters
    ----------
    parameters : list of theano variables
        to be reparameterized
    f : function theano_variable -> theano_variable
    finv : function theano_variable -> theano_variable
    """
    parameters = convert(parameters, Sequence)
    if any(is_theano_proxified(y) for y in parameters):
        raise RuntimeError(
            "parameters is already proxified. It is usually not intended to proxify things twice.")
    assert all(is_clonable(param) for param in parameters), (
        "Can only flatten clonable parameters."
    )
    underlying_parameters = []
    try:
        cp_parameters = theano.function([], parameters, on_unused_input="ignore", givens=givens, mode="FAST_COMPILE")()
    except MissingInputError as e:
        warnings.warn("MissingInputs. Using symbolic version, might be considerably slower. %s" % e)
        # clone is decisive as we otherwise get an infinite reference loop
        cp_parameters = map(clone, parameters)

    for p, cp in izip(parameters, cp_parameters):
        underlying_p = model_to_output(finv(cp))
        underlying_p.name = p.name + "_" + f.func_name  # naming is not needed if f, finv are Models
        new_p = model_to_output(f(underlying_p))
        new_p.name = (p.name or str(p)) + "_reparam"
        theano_proxify(p, new_p)
        underlying_parameters.append(underlying_p)
    return underlying_parameters


def prox_center(parameters, givens={}):
    """ centers the parameters by proxifying, returning the new underlying parameters

    Tries to compute numeric value of the parameters, however if this is not possible, a symbolic copy is used. """
    parameters = convert(parameters, Sequence)
    if any(is_theano_proxified(y) for y in parameters):
        raise RuntimeError(
            "parameters is already proxified. It is usually not intended to proxify things twice.")
    try:
        copies = theano.function([], parameters, givens=givens)()
    except MissingInputError as e:
        warnings.warn("MissingInputs. Using symbolic version, might be considerably slower. %s" % e)
        assert all(is_clonable(p) for p in parameters), "can only center clonable parameters"
        copies = [clone(p) for p in parameters]

    # this works for both numeric or symbolic "copies"
    zeros = [T.zeros(cp.shape) for cp in copies]
    for z, p in izip(zeros, parameters):
        z.name = str(p) + "_centered"
    for p, z, cp in izip(parameters, zeros, copies):
        new_p = z + cp
        new_p.name = str(p) + "_centered"
        theano_proxify(p, new_p)
    return zeros


def prox_flatten(parameters, givens={}):
    """ flattens given parameters by using proxify, returning the new underlying flattend vector

    if the shape information cannot be numericalized, a symbolic copy is used for inferring symbolic shapes """
    try:
        if not isinstance(parameters, Sequence):
            raise ValueError("`parameters` is not Sequence. Nothing to flat.")
        if any(is_theano_proxified(y) for y in parameters):
            raise RuntimeError(
                "parameters is already proxified. It is usually not intended to proxify things twice.")
        flat_sym = T.concatenate([p.flatten() for p in parameters])
        shapes_sym = [p.shape for p in parameters]
        _f = theano.function([], [flat_sym] + shapes_sym, on_unused_input="warn", givens=givens, mode="FAST_COMPILE")()
        flat_num, shapes = _f[0], _f[1:]
        flat = as_tensor_variable(flat_num)
        sizes = map(np.prod, shapes)

    except MissingInputError as e:
        warnings.warn("MissingInputs. Using symbolic version, might be considerably slower. %s" % e)
        assert all(is_clonable(p) for p in parameters), "can only flatten clonable parameters"
        # it is unfortunately not trivial how to flatten parameters
        # one crucial thing is to handle interdependencies of parameters, meaning that p3 could depend on p1
        # while both are parameters finally. If p3 comes before p1, we get that
        # p3? -> flat[p3_slice]? -> p3_cp.shape? -> p1? -> flat[p1_slice]? -> p3_cp.shape?
        # where the last is indeed a pTHREE_cp.shape because p1 comes after p3 and hence needs also p3's shape
        # to get its position in the flat string
        # Fortunately, we can assume that there is no cyclic dependency between parameters as between any
        # well formed theano variables. It is tree-like orderable.

        copies = clone_all(parameters)
        flat = T.concatenate([cp.flatten() for cp in copies])
        shapes = [cp.shape for cp in copies]
        sizes = [cp.size for cp in copies]

    # CAUTION: this requires extra escaping that this works with d3viz and graphviz, because colon : in names has extra semantics
    # see http://stackoverflow.com/questions/31523810/pydot-error-involving-parsing-character-followed-by-number
    flat.name = '"%s"' % ':'.join((p.name or str(p)) for p in parameters)
    i = 0
    for p, size, shape in izip(parameters, sizes, shapes):
        new_p = flat[i:i + size].reshape(shape)
        new_p.name = (p.name or str(p)) + "_flat"
        theano_proxify(p, new_p)
        i += size
    return flat


"""
Concrete Helper Models
----------------------
"""

"""
for positive values:
"""
# eps = as_tensor_variable(1e-16)
eps = as_tensor_variable(1e-9)

def softplus(x, module=T):
    return module.log(module.exp(x) + 1)

def softplus_inv(y, module=T):
    return module.log(module.exp(y) - 1)

def squareplus(x, module=T):
    return module.square(x) + eps  # to ensure >= 0

def squareplus_inv(x, module=T):
    return module.sqrt(x - eps)


"""
for p-values (0,1)
"""

def tan_01_R(x):
    return T.tan(np.pi * (x - 0.5))

def tan_01_R_inv(y):
    return T.ifelse(T.lt(y, 0), -(T.arctan(T.inv(y))) / np.pi, 1-(T.arctan(T.inv(y))) / np.pi)

def square_01_R(x):
    return (2*x - 1) / (x - x*x)

def square_01_R_inv(y, module=T):
    return (module.sqrt(y*y + 4) + y - 2) / (2*y)


def logit(x):
    return -T.log(T.inv(x) - 1)

def logistic(y):
    T.inv(1 + T.exp(-y))


"""
psumto1
"""
def softmax(y, module=T):
    expy = module.exp(y)
    return expy / expy.sum()


def softmax_inv(x, initial_normalization=1, module=T):
    return module.log(x*initial_normalization)


"""
norms and distances
-------------------
"""

def L1(parameters):
    parameters = convert(parameters, Sequence)
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += abs(p).sum()
    return summed_up / n


def L2(parameters):
    parameters = convert(parameters, Sequence)
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += (p**2).sum()
    return summed_up / n


def norm_distance(targets, outputs, norm=L2):
    """ targets and outputs are assumed to be *lists* of theano variables """
    return norm([t - o for t, o in izip(targets, outputs)])


"""
reshape helpers
---------------
"""

def total_size(variables):
    """ clones by default, as this function is usually used when something is meant to be replaced afterwards """
    variables = convert(variables, Sequence)
    try:
        sizes = theano.function([], [v.size for v in variables], mode="FAST_COMPILE")()
    except MissingInputError:
        sizes = [clone(v).size for v in variables]
    return reduce(op.add, sizes)  # for generality to also include numerical sizes


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
                yield Model(track=False, inputs=inputs, outputs=outputs)
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





