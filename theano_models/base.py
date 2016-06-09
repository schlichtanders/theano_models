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

from util import clone, as_tensor_variable, deepflatten_keep_vars
from util.theano_helpers import is_clonable, get_inputs
import types

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Core Components
===============

list of input/output reference names
------------------------------------
As all references will either go into the graph or out, (or only helpers), we summarize them:
"""

inputting_references = set(['inputs'])
outputting_references = set(['outputs'])


"""
decorator helper
----------------
"""


def model_to_output(m):
    if isinstance(m, Sequence) and any(isinstance(n, Model) for n in m):
        return map(model_to_output, m)
    elif isinstance(m, Model):
        return m['outputs']
    else:
        return m


@wrapt.decorator
def models_as_outputs(wrapped, instance, args, kwargs):
    return wrapped(*map(model_to_output, args), **fmap(model_to_output, kwargs))


"""
core class
----------
"""


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

    ALLOWED_VALUETYPES = gof.Variable, types.FunctionType
    current_model = 1
    all_models = []

    @models_as_outputs
    def __init__(self, outputs, inputs=None, name=None, **further_references):
        """
        Constructs a Model by directly referencing outputs and inputs, and further_references
        It does essentially nothing, but gives them a summarizing class interface.

        Parameters
        ----------
        outputs : list of theano expressions (or exceptionally also single expression)
            functional like outputs
        inputs: list of theano expressions (or exceptionally also single expression)
            functional like inputs
            if not given explicitly ``inputs`` get set to ``theano_models.util.theano_helpers.get_inputs(outputs)``
        further_references: kwargs of string: (theano expressions, or lists thereof)
            possible further references
        """
        if inputs is not None and not isinstance(inputs, Sequence):
            warnings.warn("Detected singleton input and wrapped it to ``inputs = [input]``.")
            inputs = [inputs]

        if name is None:
            self.name = "%s%i" % (self.__class__.__name__, self.__class__.current_model)
            self.__class__.current_model += 1
        else:
            self.name = name

        _outputs = outputs if isinstance(outputs, Sequence) else [outputs]
        # set names explicitly
        for idx, o in enumerate(_outputs):
            if o.name is None:
                o.name = "%s.%i" % (self.name, idx)
        for idx, i in enumerate(inputs):
            if i.name is None:
                i.name = "%s.inputs.%i" % (self.name, idx)

        if inputs is None:
            inputs = get_inputs(_outputs)
        self.references = {
            'outputs': outputs,
            'inputs': inputs,
        }
        self.references.update(further_references)
        Model.all_models.append(self)

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
                pass

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
            old_cp = [clone(o) for o in
                      old]  # we need copy as new gets proxified later on, single copy satifies as this is not recursive
            new = complex_reshape(new, old_cp)  # list(...) not needed because of @track_as_helper decorator
        else:
            singleton = True
            new = [new]

        # check args
        # ==========

        assert len(old) == len(new), "No substitution as length of `self[%s]` (%s) and `new`(%s) differ" % (
            key, len(old), len(new))
        # if len(old) != len(new):
        #     raise TypeError()
        #     warnings.warn("No substitution as length of `self[key]` (%s) and `new`(%s) differ. Key is simply replaced." % (len(old), len(new)))
        #     self.references[key] = new[0] if singleton else new
        #     return

        # the only exception to substitution is if other variables are stored
        # which by default guarantee correct referencess (e.g. Functions)
        if any(not isinstance(o, theano.gof.Variable)
               and isinstance(n, self.ALLOWED_VALUETYPES)
               for o, n in zip(old, new)):
            info = "No substitution as `self[%s]` is not a theano.gof.Variable. Key is simply replaced." % key
            print info  # warnings.warn(info)
            self.references[key] = new[0] if singleton else new
            return

        assert all(o.type == n.type for o, n in izip(old, new)), "No substitution as length theano types differ"

        # core substitution
        # =================
        for o, n in izip(old, new):
            proxify(o, n)
        # make sure that simply all cached compiled functions get destroyed, as references are no longer valid
        reset_eval(self)

    @models_as_outputs
    def __setitem__(self, key, value):
        """ convenience access to substitute_key """
        if key in self:
            self.substitute_key(key, value)
        else:
            if not isinstance(value, self.ALLOWED_VALUETYPES):
                raise TypeError(
                    "The type of the given value is not supported. You may change ``Model.ALLOWED_VALUETYPES`` if you know what your doing.")
            self.references[key] = value

    # #deprecated as the call might once work like in theano.OpFromGraph
    # def __call__(self, *inputs):
    #     """ CAUTION: works inplace
    #
    #     Parameters
    #     ----------
    #     inputs : list of input
    #         must match self['inputs'] theano expressions
    #
    #     Returns
    #     -------
    #     outputs
    #     """
    #     self['inputs'] = inputs
    #     return self['outputs']


    # Mappings Interface
    # ------------------

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
Merge helpers
=============
"""


def reparameterize(parameters, f, finv):
    """
    use e.g. within a Merge
    >>> reparameterize(model['parameters_positive'], softplus, softplusinv)
    to get new parameters

    new_param = finv(param)
        param = f(new_param)

    Parameters
    ----------
    parameters : list of theano variables
        to be reparameterized
    f : function theano_variable -> theano_variable
    finv : function theano_variable -> theano_variable

    Returns
    -------
    new underlying parameters
    (i.e. NOT the reparameterized parameters, they are substituted, i.e. references still hold)
    """
    assert all(is_clonable(param) for param in parameters), (
        "Can only flatten clonable parameters."
    )
    new_underlying_parameters = []
    for param in parameters:
        cp_param = clone(param)
        cp_param.name = (cp_param.name or str(cp_param))  # + "_copy"
        new_param = model_to_output(finv(cp_param))  # clone is decisive as we otherwise get an infinite reference loop
        new_param.name = cp_param.name + "_" + f.func_name  # naming is not needed if f, finv are Models
        proxified_param = model_to_output(f(new_param))
        proxified_param.name = (param.name or str(param)) + "_reparam"
        proxify(param, proxified_param)
        new_underlying_parameters.append(new_param)
    return new_underlying_parameters


def merge_key(models, key="parameters"):
    """ simply combines all model[key] values for model in models """
    parameters = []
    for g in models:
        if key in g:
            parameters += g[key]
    # return [p for p in parameters if isinstance(p, SharedVariable)]
    return parameters  # no filtering for SharedVariable possible as everything is theano variable (maybe constant variable)


def merge_key_reparam(models, f, finv, key="parameters", key_reparam="parameters_positive"):
    return merge_key(models, key) + reparameterize(merge_key(models, key_reparam), f, finv)


def pmerge_key_reparam(f, finv, **outer_kwargs):
    def _merge_key_reparam(models, **inner_kwargs):
        update(inner_kwargs, outer_kwargs, overwrite=False)
        return merge_key_reparam(models, f, finv, **inner_kwargs)
    return _merge_key_reparam

    return merge_key(models, key) + reparameterize(merge_key(models, key_reparam), f, finv)


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

    all_merges = []

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
        Merge.all_merges.append(self)



"""
Vizualization Helper Model
==========================
"""


class Helper(Model):
    all_helpers = []

    def __init__(self, *args, **kwargs):
        Helper.all_helpers.append(self)
        super(Helper, self).__init__(*args, **kwargs)


@wrapt.decorator
def track_as_helper(wrapped, instance, args, kwargs):
    outputs = wrapped(*args, **kwargs)
    if isinstance(outputs, types.GeneratorType):
        outputs = list(outputs)
    kwargs['inputs'] = deepflatten_keep_vars(args)
    Helper(outputs=outputs, name=wrapped.func_name, **kwargs)  # this is just for bookkeeping
    return outputs


"""
reparameterization helpers
--------------------------
"""

eps = as_tensor_variable(0.0001)

@track_as_helper
def softplus(x, module=T):
    return module.log(module.exp(x) + 1)

@track_as_helper
def softplus_inv(y, module=T):
    return module.log(module.exp(y) - 1)

@track_as_helper
def squareplus(x, module=T):
    return module.square(x) + eps  # to ensure >= 0

@track_as_helper
def squareplus_inv(x, module=T):
    return module.sqrt(x - eps)


"""
norms and distances
-------------------
"""

@track_as_helper
def L1(parameters):
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += abs(p).sum()
    return summed_up / n

@track_as_helper
def L2(parameters):
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += (p**2).sum()
    return summed_up / n


def norm_distance(norm=L2):
    @track_as_helper
    def distance(targets, outputs):
        """ targets and outputs are assumed to be *lists* of theano variables """
        return norm([t - o for t, o in izip(targets, outputs)])
    return distance


"""
reshape helpers
---------------
"""

@track_as_helper
def total_size(variables):
    """ clones by default, as this function is usually used when something is meant to be replaced afterwards """
    if not isinstance(variables, Sequence):
        variables = [variables]
    return sum(clone(v).size for v in variables)

@track_as_helper
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

