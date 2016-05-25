#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

import warnings
from collections import Sequence, MutableMapping
from copy import copy
from itertools import izip
from pprint import pformat
import wrapt
from functools import partial

import numpy as np
import theano
import theano.tensor as T
from theano import gof, config
from theano.compile.sharedvalue import SharedVariable

from schlichtanders.mydicts import update
from schlichtanders.mylists import sequencefy, remove_duplicates
from schlichtanders.mymeta import proxify
from schlichtanders.myfunctools import fmap

from util import complex_reshape, clone, as_tensor_variable
from util.theano_helpers import is_clonable

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
        if inputs is not None and not isinstance(inputs, Sequence):
            warnings.warn("Detected singleton input and wrapped it to ``inputs = [input]``.")
            inputs = [inputs]

        self.references = {
            'outputs': outputs,
            'inputs': gof.graph.inputs(outputs) if inputs is None else inputs,
        }
        self.references.update(further_references)

    def __copy__(self):
        cls = self.__class__
        cp = cls.__new__(cls)
        cp.references = {k: copy(v) for k, v in self.references.iteritems()}
        return cp

    def function(self, *args, **kwargs):
        if 'on_unused_input' not in kwargs:
            kwargs['on_unused_input'] = "warn"
        return theano.function(self['inputs'], self['outputs'], *args, **kwargs)

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
        elif not isinstance(new, Sequence):
            try:
                new = as_tensor_variable(new)
            except TypeError:
                # default to direct assignment
                warnings.warn("non theano value, overwriting key directly")
                self.references[key] = new
                return

        if isinstance(new, Sequence):
            # if there are models, recognize them and use there outputs
            # no fancy rewriting here, as the mapping is ambigous
            _new = []
            for n in new:
                if isinstance(n, Model):
                    _new.extend(sequencefy(n['outputs']))
                else:
                    _new.append(as_tensor_variable(n))
            new = _new
        # no Sequence, try FANCY REWRITING
        elif (hasattr(new, 'broadcastable') and new.broadcastable == (False,)
              and all(is_clonable(o) for o in old)):  # vector type of arbitrary dtype
            print "fancy reshaping"
            old_cp = [clone(o) for o in old]  # we need copy as new gets proxified later on, single copy satifies as this is not recursive
            new = list(complex_reshape(new, old_cp))
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
        # make sure that simply all cached compiled functions get destroyed, as references are no longer valid
        reset_eval(self)

    def __setitem__(self, key, value):
        """ convenience access to substitute_key """
        if key in self:
            self.substitute_key(key, value)
        else:
            if isinstance(value, Model):
                value = value['outputs']
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
            try:
                self[append_key] += new
            except KeyError:
                self[append_key] = new

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
Caution with eval()
-------------------
"""


def reset_eval(var):
    """ this empties the caches of compiled functions

    Parameters
    ----------
    var : Model, Sequence, or theano Variable
        to be reset (maybe recursively)
    """
    if isinstance(var, Model):
        for key in var:
            reset_eval(var[key])
    elif isinstance(var, Sequence):
        for subvar in var:
            reset_eval(subvar)
    elif isinstance(var, gof.Variable):
        if hasattr(var, '_fn_cache'):
            del var._fn_cache
    # everything else does not need to be reset


"""
decorator helper
----------------
"""

@wrapt.decorator
def models_as_outputs(wrapped, instance, args, kwargs):
    def to_output(m):
        return m['outputs'] if isinstance(m, Model) else m
    return wrapped(*map(to_output,args), **fmap(to_output, kwargs))

"""
Merge helpers
-------------
"""


def merge_key(models, key="parameters"):
    """ simply combines all model[key] values for model in models """
    parameters = []
    for g in models:
        if key in g:
            parameters += g[key]
    # return [p for p in parameters if isinstance(p, SharedVariable)]
    return parameters  # no filtering for SharedVariable possible as everything is theano variable (maybe constant variable)


def merge_inputs(models, key="inputs"):
    """ combines all inputs, retaining only such with empty owner """
    inputs = []
    for g in models:
        inputs += g[key]
    return [i for i in inputs if i.owner is None]


class Merge(Model):
    """ This class is merely for convenience and suggestion.

    simple manual version::
    >>> merged_ = {k:merge_key(models, k) for k in ('parameters', 'parameters_positive', 'inputs'}
    >>> merged = Model(**update(merged_, models[0], overwrite=False)
    or with alternativ ending
    >>> merged = Model(**update(dict(models[0]), merged_))
    which is shorter, but with slight copy overhead.


    """
    def __init__(self, *models, **merge_rules):
        """
        inputs, parameters and parameters_positive are merged by default if not overwritten in merge_rules
        first model is regarded as Like model

        If you don't want this behaviour consider using Model directly to create new models.

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
        update(merge_rules, {
                'parameters': merge_key,
                'parameters_positive': partial(merge_key, key="parameters_positive"),
                'inputs': merge_inputs,
            }, overwrite=False)

        merged_references = {}
        for k, m in merge_rules.iteritems():
            if hasattr(m, '__call__'):
                merged_references[k] = m(models)
            else:
                merged_references[k] = m

        for m in merged_references.itervalues():
            if isinstance(m, Sequence):
                remove_duplicates(m)

        update(merged_references, models[0], overwrite=False)
        super(Merge, self).__init__(**merged_references)


