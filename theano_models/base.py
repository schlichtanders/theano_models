#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

import warnings
from collections import Sequence, MutableMapping
from copy import copy
from itertools import izip
import json

import numpy as np
import theano
import theano.tensor as T
from theano import gof, shared, config
from theano.compile.sharedvalue import SharedVariable

from schlichtanders.mydicts import update
from schlichtanders.myfunctools import Compose, I, fmap
from schlichtanders.mylists import sequencefy
from schlichtanders.mymeta import proxify
from util.theano_helpers import complex_reshape, SymbolicSharedVariable
from pprint import pformat

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


class Model(MutableMapping):
    """
    This structure is intended to be a view onto a TheanoGraph with extra references to:
        - either theano variables directly,
        - or MutableSequence of theano variables (a MutableSequence with nameable entries
          can be found in schlichtanders.mylists.DictList)
        (These references are directly accessible through dictionary interface.)

    The two basic references are 'inputs' and 'outputs'. This mirrors the core idea of this class, nameley
    to have a more abstract equivalent to theano.function(inputs, outputs).
    With this, Graphs are intended to be composable, namely by mapping ``graph1['outputs'] -> graph2['inputs']``
    and so on.
    """

    def __init__(self, outputs, inputs=None, **further_references):
        """
        Constructs a Graph by directly referencing outputs and inputs, and further_references
        It does essentially nothing, but gives them a summarizing class interface.

        Parameters
        ----------
        outputs : list of theano expressions (or exceptionally also single expression)
            functional like outputs
        inputs: list of theano expressions (or exceptionally also single expression)
            functional like inputs
            if not given explicitly ``inputs`` get set to ``theano.gof.graph.inputs(outputs)``

        further_references: kwargs of string: (theano expressions, or lists thereof)
            possible further references
        """
        if isinstance(outputs, Model):
            outputs = outputs['outputs']
        if isinstance(inputs, Model):
            inputs = Model['outputs']
        if not isinstance(inputs, Sequence):
            raise ValueError("need *list* of theano expressions as inputs")

        self.references = {
            'outputs': outputs,
            'inputs': gof.graph.inputs(outputs) if inputs is None else inputs,
        }
        self.references.update(further_references)
        self._postmap = I  # we use a complex Compose, so that we can call the _postmap with complex kwargs

    def __copy__(self):
        cls = self.__class__
        cp = cls.__new__(cls)
        cp.references = {k: copy(v) for k, v in self.references.iteritems()}
        return cp

    def function(self):
        return theano.function(self['inputs'], self['outputs'])

    def postmap(self, **kwargs):
        return self._postmap(self, **kwargs)

    def add_postmap(self, postmap, call_first=False):
        """ combines existing postmap with new postmap """
        if call_first:
            self._postmap = self._postmap + postmap
        else:
            self._postmap = postmap + self._postmap

    # Substitution Interface
    # ----------------------

    def substitute_key(self, key, new):
        """ substitutes self[key] with new

        if they match in len and type, the old variables will get proxified to follow the new variables
        """
        # Preprocessing
        # =============
        old = self[key]
        singleton = False
        if not isinstance(old, Sequence):
            old = [old]

        if isinstance(new, Model):
            new = new['outputs']

        if isinstance(new, Sequence):
            # if there are graphs, recognize them and use there outputs
            # no fancy rewriting here, as the mapping is ambigous
            _new = []
            for n in new:
                if isinstance(n, Model):
                    _new.extend(sequencefy(n['outputs']))
                else:
                    _new.append(n)
            new = _new
        # no Sequence, try FANCY REWRITING
        elif hasattr(new, 'broadcastable') and new.broadcastable == (False,):  # vector type of arbitrary dtype
            new = list(complex_reshape(new, old))
        else:
            singleton = True
            new = [new]


        # check args
        # ==========
        if len(old) != len(new):
            warnings.warn("No substitution as length of `self[key]` (%s) and `new`(%s) differ. Key is simply replaced." % (len(old), len(new)))
            self.references[key] = new[0] if singleton else new
            return
        if not isinstance(old[0], theano.gof.Variable):  # TODO assumption that a key has uniform class
            warnings.warn("No substitution as `self[key]` is not a theano.gof.Variable. Key is simply replaced.")
            self.references[key] = new[0] if singleton else new
            return
        assert all(o.type == n.type for o, n in izip(old, new))

        # core substitution
        # =================
        for o, n in izip(old, new):
            proxify(o, n)

    def __setitem__(self, key, value):
        """ convenience access to substitute_key """
        if key in self:
            self.substitute_key(key, value)
        else:
            self.references[key] = value

    def __call__(self, *inputs):
        """ CAUTION: works inplace

        Parameters
        ----------
        inputs : list of input
            must match self['inputs'] theano expressions

        Returns
        -------
        outputs
        """
        self['inputs'] = inputs
        return self['outputs']

    def map(self, key, f, append_key=None):
        """ adds ``f(var)`` to ``self[append_key]``  for each ``var in self[key]``

        if ``append_key`` is None, then the results won't be added anywhere, only f is called
        """
        new = map(f, self[key])
        if append_key is not None:
            self[append_key] += new

    # Mappings Interface
    # ------------------

    def __getitem__(self, item):
        return self.references[item]

    def __delitem__(self, key):
        del self.references[key]

    def __iter__(self):
        return iter(self.references)

    def __len__(self):
        return len(self.references)

    # visualization interface
    # -----------------------

    def __str__(self):
        def str_value(v):  # because str(list) does call repr() internally
            return map(str, v) if isinstance(v, Sequence) else str(v)
        return json.dumps(fmap(str_value, self), sort_keys=True, indent=2)

        return pformat(
            '{%s}' % ','.join('%s: %s' % (k, str_v(v)) for k, v in self.iteritems()),
            indent=2
        )

    def __repr__(self):
        return pformat(
            '{%s}' % ','.join('%s: %s' % (k, repr(v)) for k, v in self.iteritems()),
            indent=2
        )


"""
Merge
=====
"""


def merge_parameters(graphs, key="parameters"):
    """ combines all params, retaining only SharedVariables """
    parameters = []
    for g in graphs:
        parameters += g[key]
    return [p for p in parameters if isinstance(p, SharedVariable)]


def merge_inputs(graphs, key="inputs"):
    """ combines all inputs, retaining only such with empty owner """
    inputs = []
    for g in graphs:
        inputs += g[key]
    return [i for i in inputs if i.owner is None]


def merge(model_type, *graphs, **kwargs):
    """ merges references of ``graph`` into ``self`` (cares about duplicates)

    This method is only for convenience and suggestion.
    Short version::
    >>> merged_ = {'parameters':merge_parameters(graphs), 'inputs':merge_inputs(graphs)}
    >>> merged = Model(**update(merged_, graphs[0], overwrite=False)
    or with alternativ ending
    >>> merged = Model(**update(dict(graphs[0]), merged_))
    which is shorter, but with slight copy overhead.

    Parameters
    ----------
    model_type : model class
        initialized with kwargs
    graphs : list of Graph
        used for further merging
        first graph is regarded as base graph which additional keys will be used
    merge_rules: dictionary of functions working on graphs
        mapping merge_key to merger
    """
    merge_rules = kwargs.pop('merge_rules', {'parameters':merge_parameters, 'inputs':merge_inputs})  # python 3 keyword arg alternative

    merged = {k:v(graphs) for k, v in merge_rules.iteritems()}
    update(merged, graphs[0], overwrite=False)

    return model_type(**merged)



"""
reparameterization
==================
"""


def reparameterize_map(f, finv):
    """
    use e.g. like
    >>> graph.map("parameters_positive", reparameterize_map(softplus, softplusinv), "parameters")

    Parameters
    ----------
    f : function
        only theano.tensor functions are needed here
    finv : function
        should have module kwarg indicating where the methods should be used from (numpy or theano.tensor)
    Returns
    -------
        mappable function
    """
    def reparameterize(sv):
        if isinstance(sv, SymbolicSharedVariable):
            new_sv = shared(finv(sv.symbol, module=T))
        else:
            new_sv = shared(finv(sv.get_value(), module=np))
        proxify(sv, f(new_sv))
        return new_sv  # instead of old parameter, now refer directly to the new underlying parameter
    return reparameterize


def flatten_parameters(graph):
    values = [p.get_value(borrow=True) for p in graph['parameters']]
    values_flat = np.empty((sum(v.size for v in values),))
    parameters_flat = shared(values_flat, dtype=config.floatX, borrow=True)
    i = 0
    for p, v in zip(graph['parameters'], values):
        values_flat[i:i + v.size] = v.flat
        new_p = parameters_flat[i:i + v.size].reshape(v.shape)
        proxify(p, new_p)  # p.set_value(, borrow=True)
        i += v.size

    del graph['parameters']
    graph['parameters'] = [parameters_flat]