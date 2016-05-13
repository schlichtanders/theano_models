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
from theano import gof, config
from theano.compile.sharedvalue import SharedVariable

from schlichtanders.mydicts import update
from schlichtanders.myfunctools import Compose, I, fmap
from schlichtanders.mylists import sequencefy, remove_duplicates
from schlichtanders.mymeta import proxify
from util import complex_reshape, shared
from util.theano_helpers import symbolic_shared_variables
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
    With this, Models are intended to be composable, namely by mapping ``model1['outputs'] -> model2['inputs']``
    and so on.
    """

    def __init__(self, outputs, inputs=None, **further_references):
        """
        Constructs a Model by directly referencing outputs and inputs, and further_references
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

    def set_postmap(self, postmap):
        self._postmap = Compose(postmap)

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
            # if there are models, recognize them and use there outputs
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
            print "fancy reshaping"
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
        return pformat(dict(self), indent=2)

    __repr__ = __str__


"""
Merge
=====
"""


def merge_parameters(models, key="parameters"):
    """ combines all params, retaining only SharedVariables """
    parameters = []
    for g in models:
        parameters += g[key]
    return [p for p in parameters if isinstance(p, SharedVariable)]


def merge_inputs(models, key="inputs"):
    """ combines all inputs, retaining only such with empty owner """
    inputs = []
    for g in models:
        inputs += g[key]
    return [i for i in inputs if i.owner is None]


def merge(model_type, *models, **kwargs):
    """ This method is only for convenience and suggestion.
    Short version::
    >>> merged_ = {'parameters':merge_parameters(models), 'inputs':merge_inputs(models)}
    >>> merged = Model(**update(merged_, models[0], overwrite=False)
    or with alternativ ending
    >>> merged = Model(**update(dict(models[0]), merged_))
    which is shorter, but with slight copy overhead.

    Parameters
    ----------
    model_type : model class
        initialized with kwargs
    models : list of Model
        used for further merging
        first model is regarded as base model which additional keys will be used
    merge_rules: dictionary of functions working on models
        mapping merge_key to merger
    """
    merge_rules = kwargs.pop('merge_rules', {'parameters': merge_parameters, 'inputs': merge_inputs})  # python 3 keyword arg alternative

    merged = {k: m(models) for k, m in merge_rules.iteritems()}
    update(merged, models[0], overwrite=False)
    fmap(remove_duplicates, merged)
    return model_type(**merged)



"""
reparameterization
==================
"""


def reparameterize_map(f, finv):
    """
    use e.g. like
    >>> model.map("parameters_positive", reparameterize_map(softplus, softplusinv), "parameters")

    Parameters
    ----------
    f : function
        only theano.tensor functions are needed here
    finv : function with module kwarg
        module kwarg must indicate where the methods should be used from (numpy or theano.tensor typically)

    Returns
    -------
        mappable function
    """
    def reparameterize(sv):
        if sv in symbolic_shared_variables:
            ret = shared(finv(symbolic_shared_variables[sv], module=T))
        else:
            ret = shared(finv(sv.get_value(), module=np))
        ret.name = sv.name + "_" + f.func_name
        new_sv = f(ret)
        new_sv.name = sv.name
        proxify(sv, new_sv)
        return ret  # instead of old parameter, now refer directly to the new underlying parameter
    return reparameterize


def flatten_parameters(model):
    names = [p.name if p.name is not None else "_" for p in model['parameters']]
    values = [p.get_value(borrow=True) for p in model['parameters']]  # borrow=True is only faster
    values_flat = np.empty((sum(v.size for v in values),))
    parameters_flat = shared(values_flat, borrow=True)
    i = 0
    for p, v in zip(model['parameters'], values):
        values_flat[i:i + v.size] = v.flat
        new_p = parameters_flat[i:i + p.size].reshape(p.shape)
        proxify(p, new_p)  # p.set_value(, borrow=True)
        i += v.size
    parameters_flat.name = ":".join(names)
    del model['parameters']
    model['parameters'] = [parameters_flat]