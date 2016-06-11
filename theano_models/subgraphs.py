#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import warnings
from collections import Sequence, MutableMapping, Mapping, defaultdict
from copy import copy
from itertools import izip
from pprint import pformat
import wrapt
from functools import wraps
import theano.tensor as T
from schlichtanders.mydicts import update
from schlichtanders.myfunctools import fmap, convert
from schlichtanders.mylists import remove_duplicates, shallowflatten

from util import clone, as_tensor_variable, deepflatten_keep_vars, get_unique_name, shallowflatten_keep_vars
from util.theano_helpers import is_clonable, get_inputs
import types

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
list of input/output reference names
------------------------------------
As all references will either go into the graph or out, (or only helpers), we summarize them:
"""
inputting_references = set(['inputs'])
outputting_references = set(['outputs'])


"""
Subgraph class
--------------
"""


class Subgraph(MutableMapping):  # == HashableDict with unique name

    all_subgraphs = []

    def __init__(self, dict_like=None, name=None, ignore=False, **kwargs):
        if isinstance(dict_like, Subgraph):
            self.references = dict_like.references  # to prevent unnecessary nestings
        elif dict_like is None:
            self.references = {}
        else:
            self.references = dict_like

        self.references.update(kwargs)

        if name is None:
            name = self.__class__.__name__
        self.name = get_unique_name(name)

        # set names of references if not done so already
        for k, v in self.iteritems():
            if isinstance(v, Sequence):
                for i, x in enumerate(v):
                    if hasattr(x, 'name') and x.name is None:
                        x.name = "%s.%s.%i" % (self.name, k, i)
            else:
                if hasattr(v, 'name') and v.name is None:
                    v.name = "%s.%s" % (self.name, k)

        if not ignore:
            Subgraph.all_subgraphs.append(self)

    def __copy__(self):
        cls = self.__class__
        cp = cls.__new__(cls)
        cp.references = {k: v[:] if isinstance(v, Sequence) else v for k, v in self.references.iteritems()}
        return cp

    # dict interface
    # --------------
    def __setitem__(self, key, value):
        self.references[key] = value

    def __getitem__(self, item):
        return self.references[item]

    def __delitem__(self, key):
        # TODO should delete be allowed? seems like producing bugs in that references can be deleted and added anew, e.g. with different lengths
        del self.references[key]

    def __iter__(self):
        return iter(self.references)

    def __len__(self):
        return len(self.references)

    # hashable interface
    # ------------------

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return hash(self) == hash(other)


    # visualization interface
    # -----------------------

    def __str__(self):
        return self.name + " " + pformat(dict(self), indent=2)

    __repr__ = __str__


class ModifySubgraph(Subgraph):
    NONE = "__NONE__"

    def __init__(self, base_dict, name=None, ignore=False, modify_getitem=lambda key, value: value):
        self.modify_getitem = modify_getitem
        if hasattr(base_dict, 'name'):
            if name is None:
                name = base_dict.name
            else:
                name = base_dict.name + "." + name
        super(ModifySubgraph, self).__init__(base_dict, name, ignore)

    def __getitem__(self, item):
        ret = self.modify_getitem(item, self.references.get(item, self.NONE))
        if ret == self.NONE:
            raise KeyError("Key %s not found." % item)
        return ret


"""
decorator helpers
-----------------
"""


def subgraph_to_output(m):
    if isinstance(m, Sequence) and any(isinstance(n, Subgraph) for n in m):
        return shallowflatten_keep_vars(map(subgraph_to_output, m))
    elif isinstance(m, Subgraph):
        return m['outputs']
    else:
        return m


@wrapt.decorator
def subgraphs_as_outputs(wrapped, instance, args, kwargs):
    return wrapped(*map(subgraph_to_output, args), **fmap(subgraph_to_output, kwargs))


@subgraphs_as_outputs
def subgraph(*extra_inputs, **extra_references):
    """ Decorates a function to be listed as Subgraph with optional extra inputs references """
    extra_inputs = deepflatten_keep_vars(list(extra_inputs) + list(extra_references.pop('inputs', [])))

    @wrapt.decorator
    def decorator(wrapped, instance, args, kwargs):
        outputs = wrapped(*args, **kwargs)
        if isinstance(outputs, types.GeneratorType):
            outputs = list(outputs)

        Subgraph(dict(
            inputs=remove_duplicates(extra_inputs + deepflatten_keep_vars(args)),
            outputs=outputs,
            **extra_references
        ), wrapped.func_name)
        return outputs
    return decorator


def subgraph_modify(dict_like):
    """ Decorates a function to be listed as ModifySubgraph over ``proxy`` with optional extra inputs references """
    def decorator(wrapped):
        name = (dict_like.name + ".") if hasattr(dict_like, 'name') else ""
        name += wrapped.func_name
        base_subgraph = Subgraph(dict_like, name, ignore=True)  # this should not be listed explicitly, it is only bookkeeping
        wrapped.base_subgraph = base_subgraph

        @wraps(wrapped)
        def wrapper(*args, **kwargs):
            outputs = wrapped(*args, **kwargs)
            if isinstance(outputs, types.GeneratorType):
                outputs = list(outputs)

            def append_to(key, val):
                if key == 'outputs':
                    return outputs  # no extra references here, no val
                elif key == 'inputs':
                    return remove_duplicates(val + list(args))
                return val

            ModifySubgraph(base_subgraph, modify_getitem=append_to)
            return outputs
        return wrapper
    return decorator


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
    if not isinstance(variables, Sequence):
        variables = [variables]
    return sum(clone(v).size for v in variables)

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

