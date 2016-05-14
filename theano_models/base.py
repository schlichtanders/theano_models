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
from schlichtanders.mylists import sequencefy, remove_duplicates
from schlichtanders.mymeta import proxify
from util import complex_reshape, clone, as_tensor_variable
from util.theano_helpers import is_clonable
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
            new = as_tensor_variable(new)

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
        elif (
            hasattr(new, 'broadcastable') and new.broadcastable == (False,)
            and all(is_clonable(o) for o in old)
        ):  # vector type of arbitrary dtype
            print "fancy reshaping"
            old_cp = [clone(o) for o in old]  # we need copy as new gets proxified later on
            new = list(complex_reshape(new, old_cp))
            # note that there is no fancy reshaping of
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