#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import warnings
from collections import Sequence, MutableMapping, Mapping
from copy import copy
from itertools import izip
from pprint import pformat

import itertools
import wrapt
from functools import partial, wraps

import numpy as np
from frozendict import frozendict

import theano
import theano.tensor as T
from theano.gof.fg import MissingInputError
from schlichtanders.mycontextmanagers import until_stopped, ignored
from theano import gof, config
from theano.compile.sharedvalue import SharedVariable

from schlichtanders.mydicts import update, ModifyDict, HashableDict
from schlichtanders.mylists import sequencefy, remove_duplicates, as_list
from schlichtanders.mymeta import proxify, Proxifier
from schlichtanders.myfunctools import fmap, convert, as_wrapper, decorator_from_fmap

from util import clone, as_tensor_variable, shallowflatten_keep_vars, deepflatten_keep_vars, U, reset_eval
from util.theano_helpers import is_clonable, get_graph_inputs, clone_all, is_pseudo_constant, unbroadcastable_to_idx, \
    broadcastable_to_idx, theano_proxify
import types

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

from theano.tensor.shared_randomstreams import RandomStreams


"""
Prelimenaries
=============
Important helpers/decorators in order to make setting up things easy.
"""


def model_to_output(m):
    if isinstance(m, Sequence) and any(isinstance(n, Model) for n in m):
        return shallowflatten_keep_vars(map(model_to_output, m))
    elif isinstance(m, Model):
        return m['outputs']
    else:
        return m

@wrapt.decorator
def models_as_outputs(wrapped, instance, args, kwargs):
    return wrapped(*map(model_to_output, args), **fmap(model_to_output, kwargs))


def fmap_model(f, *args_models, **kwargs_models):
    """ maps a function over model contexts by applying it to the outputs respectively instead of the models

    returns a Merge of model outputs of f itself if it returns models, as well as the input models.
    For a fmap_model which returns the trivial Model(inputs, outputs), please use ``as_model`` combined with
    ``models_as_outputs`` decorators
    """
    outputs = f(map(model_to_output, args_models), **fmap(model_to_output, kwargs_models))
    output_models = []
    if isinstance(outputs, Sequence):
        _outputs = []
        for o in outputs:
            if isinstance(o, Model):
                output_models.append(o)
                _outputs += convert(o['outputs'], list)
            else:
                _outputs.append(o)
        outputs = shallowflatten_keep_vars(_outputs)
    return Merge(*(output_models + list(args_models) + kwargs_models.values()),
                 outputs=outputs)


"""
Model / Merge
=============
These are the two core classes of the whole module. Model simply overwrites dict assigment with a substitution mechanism
to enable interactive chaining of models. Merge is the standard class for combining Models in a clean and straighforward
default way.

First some general convention about the keys used within a Model:
"""

inputting_references = set(['inputs', 'flat', 'parameters'])
outputting_references = set(['outputs'])


def get_inputting_references(m):
    """ returns list of variables within inputting_references
    Parameters
    ----------
    m : Model
    """
    ret = []
    for r in deepflatten_keep_vars(m[k] for k in m if k in inputting_references):
        if r.name is None and (is_pseudo_constant(r)
                               or isinstance(r, theano.gof.Constant)
                               or isinstance(r, theano.tensor.sharedvar.TensorSharedVariable)):
            continue
        ret.append(r)
    return ret


def get_outputting_references(m):
    """ returns list of variables within outputting_references
    Parameters
    ----------
    m : Model
    """
    return deepflatten_keep_vars(m[k] for k in m if k in outputting_references)


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

    ALLOWED_VALUETYPES = (gof.Variable, )

    all_models = []

    weak_identity_check = True

    @models_as_outputs
    def __init__(self, outputs, inputs=None, name=None, track=True, unique_name=True, **further_references):
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
        name : str or None
            name for this model
        track : bool
            whether the model shall be tracked in the global variable ``Model.all_models``
        further_references: kwargs of theano expressions, or lists thereof
            possible further references
        """
        # references
        # ----------
        if inputs is None:
            inputs = []  # get_graph_inputs(outputs)  # this wasn't needed until now, only throughed bugs here and there
        inputs = convert(inputs, Sequence)

        self.references = {
            'inputs': inputs,
            'outputs': outputs
        }
        self.references.update(**further_references)

        # names
        # -----
        if name is None:
            name = self.__class__.__name__
        self.name = U(name) if unique_name else name

        # set names of references if not done so already
        for k, v in self.iteritems():
            if isinstance(v, Sequence):
                for i, x in enumerate(v):
                    if hasattr(x, 'name'):
                        if x.name is None:
                            x.name = "%s.%s.%i" % (self.name, k, i)
                        # we test whether there is already a parent relationship ".", if not, add this one to create unique naming
                        elif unique_name and "." not in x.name:
                            x.name = "%s.%s" % (self.name, x.name)
            else:
                if hasattr(v, 'name'):
                    if v.name is None:
                        v.name = "%s.%s" % (self.name, k)
                    # we test whether there is already a parent relationship ".", if not, add this one to create unique naming
                    elif unique_name and "." not in v.name:
                        v.name = "%s.%s" % (self.name, v.name)

        if track:
            Model.all_models.append(self)

    def __copy__(self):
        return self.copy()

    def copy(self, track=False):
        cls = self.__class__
        cp = cls.__new__(cls)
        cp.references = {k: v[:] if isinstance(v, Sequence) else v for k, v in self.references.iteritems()}
        cp.name = self.name
        if track:
            Model.all_models.append(cp)
        return cp

    def function(self, **kwargs):
        if 'on_unused_input' not in kwargs:
            kwargs['on_unused_input'] = "warn"
        return theano.function(self['inputs'], self['outputs'], **kwargs)

    # Substitution Interface
    # ----------------------

    def substitute_key(self, key, new):
        """ substitutes self[key] with new

        if they match in len and type, the old variables will get proxified to follow the new variables
        """
        # Preprocessing
        # =============
        old = self[key]
        old_singleton = not isinstance(old, Sequence)
        old = convert(old, Sequence)

        new_singleton = False
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
        elif old_singleton:
            new_singleton = True
            new = [new]
        # DEPRECATED fancy rewriting: use Flatten instead explicitly
        # # no Sequence, no single element, try FANCY REWRITING
        # elif (hasattr(new, 'broadcastable') and new.broadcastable == (False,)
        #       and all(is_clonable(o) for o in old)):  # vector type of arbitrary dtype
        #     print("fancy reshaping")
        #     old_cp = [clone(o) for o in
        #               old]  # we need copy as new gets proxified later on, single copy satifies as this is not recursive
        #     new = complex_reshape(new, old_cp)  # list(...) not needed because of @track_as_helper decorator
        else:
            new_singleton = True
            new = [new]

        # replacement of other variables
        # ==============================

        assert len(old) == len(new), "No substitution as length of `self[%s]` (%s) and `new`(%s) differ" % (
            key, len(old), len(new))

        # the only exception to substitution is if other variables are stored
        # which by default guarantee correct referencess (e.g. Functions)
        if all(not isinstance(o, theano.gof.Variable)
               and isinstance(n, self.ALLOWED_VALUETYPES)
               for o, n in zip(old, new)):
            self.references[key] = new[0] if new_singleton else new
            return

        # substitution of theano variables
        # ================================
        # reshape constant dimension:

        # def len_min_sum(iterable):
        #     return len(iterable) - sum(iterable)
        #
        # def new_reshaped():
        #     for o, n in izip(old, new):
        #         if (o.broadcastable != n.broadcastable
        #                 and len_min_sum(o.broadcastable) == len_min_sum(n.broadcastable)):  # counts False
        #             idx = itertools.count()
        #             def broadcast_pattern():
        #                 for b in o.broadcastable:
        #                     if b:
        #                         yield 'x'
        #                     else:
        #                         yield next(idx)
        #             yield n.squeeze().dimshuffle(*broadcast_pattern())
        #         else:
        #             yield n
        #
        # def new_rebroadcasted():
        #     for o, n in izip(old, new):
        #         n = T.addbroadcast(n, *broadcastable_to_idx(o.broadcastable))
        #         n = T.unbroadcast(n, *unbroadcastable_to_idx(o.broadcastable))
        #         yield n
        #
        # original_new = new
        # new = list(new_reshaped())
        # if self.weak_identity_check:
        #     assert all(o.dtype == n.dtype for o, n in izip(old, new)), "same dtype needed"
        #     assert all(len(o.broadcastable) == len(n.broadcastable) for o, n in izip(old, new)), "same dimensionality needed"
        #     # same broadcastable dimensions are still ensured to get theano type consistency
        #     new = list(new_rebroadcasted())
        # else:
        #     assert all(o.type == n.type for o, n in izip(old, new)), "No substitution as theano types differ"
        #     # this means that also broadcastables match
        # for o, n in izip(old, new):
        #     proxify(o, n)
        # self.references[key] = original_new[0] if old_singleton else original_new

        for o, n in izip(old, new):
            theano_proxify(o, n, weak_identity=self.weak_identity_check)

        # ensure matching between models:
        self.references[key] = new[0] if old_singleton else new
        # make sure that simply all cached compiled functions get destroyed, as references are no longer valid
        reset_eval(self)

    # dict interface
    # --------------

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


class Merge(Model):
    """ this class is used for combining references of several models in a consistent manner """

    def __init__(self, *subgraphs, **other_references):
        """
        Parameters
        ----------
        subgraphs : Model or dict-like of references
            to be combined consistently
        name : str
            see Model
        track : bool
            see Model, defaults to False
        convert_singletons_to_list : Bool, default False
            if True then all single variables are wrapped into lists if combined with other subgraphs
            if False then the first reference is taken
        keep_references : list of str or "all"/None
            list of references which shall be preserved, usually `inputting_references` or similar. "all"/None will
            retain all references
            This only applies to *subgraphs, NOT to **other_references
        ignore_references : list of str or None
            opposite of keep_references
        other_references : None, str, gof.Variables, list of gof.Variables
            if is None, then the respective key is deleted
            elif keywordarg="str" then the reference is understood as a renaming  new_key=old_key (appending if already existent)
            else the old references gets REPLACED by this new reference
        """
        convert_singletons_to_list = other_references.pop("convert_singletons_to_list", False)
        keep_references = other_references.pop("keep_references", None)
        ignore_references = other_references.pop("ignore_references", None)
        if keep_references == "all":
            keep_references = None
        name = other_references.pop("name", None)
        track = other_references.pop("track", False)
        self.original_subgraphs = subgraphs

        def mycopy(dict_like):
            if isinstance(dict_like, Model):
                return copy(dict_like)
            else:
                # reimplement models copy method for dict_like things
                d = HashableDict()
                for k, v in dict_like.iteritems():
                    if isinstance(v, Sequence):
                        d[k] = copy(v)
                    else:
                        d[k] = v
                return d
        self.copied_subgraphs = map(mycopy, subgraphs)

        # remove all nested references:
        input_vars = self._gen_input_vars()
        with until_stopped:
            ivar, outer = input_vars.next()
            while True:
                matched = self._delete_corresponding_output(ivar, outer)
                ivar, outer = input_vars.send(matched)
        # merge subgraphs:
        merge = {}
        subgraph_references = set(self.copied_subgraphs[0]).union(*self.copied_subgraphs[1:])
        if keep_references is not None:
            subgraph_references = subgraph_references.intersection(keep_references)
        if ignore_references is not None:
            subgraph_references = subgraph_references.difference(ignore_references)

        for key in subgraph_references:
            for sg in self.copied_subgraphs:
                with ignored(KeyError):  # for sg[key]
                    if key not in merge:
                        merge[key] = sg[key]
                    elif isinstance(merge[key], list) or convert_singletons_to_list:
                        merge[key] = convert(merge[key], list) + convert(sg[key], list)
                    else:
                        break  # breaks for loop, i.e. key is settled)

        # merge other_references
        # first call all other_references as order is not defined on kwargs
        for k, v in other_references.iteritems():
            if hasattr(v, '__call__'):
                other_references[k] = v(merge)

        for key in other_references:
            if other_references[key] is None:
                with ignored(KeyError):
                    del merge[key]

            elif isinstance(other_references[key], basestring):
                new_key = other_references[key]
                if new_key not in merge:
                    merge[new_key] = merge[key]
                elif isinstance(merge[new_key], list) or convert_singletons_to_list:
                    merge[new_key] = convert(merge[new_key], list) + convert(merge[key], list)
                # else nothing happens as subgraphs come before other_references (by syntax)
                # in every case delete old key, as it was remapped
                del merge[key]
                # old_key = other_references[key]
                # if key not in merge:
                #     merge[key] = merge[old_key]
                # elif isinstance(merge[key], list) or convert_singletons_to_list:
                #     merge[key] = convert(merge[key], list) + convert(merge[old_key], list)
                # # else nothing happens as subgraphs come before other_references (by syntax)
                # del merge[old_key]
            else:
                with ignored(KeyError):
                    # in case merge becomes Model during code development, this could be a severe bug source
                    del merge[key]
                merge[key] = other_references[key]

        # remove all duplicates if any
        for v in merge.values():
            if isinstance(v, Sequence):
                remove_duplicates(v)  # remove_duplicates works inplace

        # deleting empty lists at the end does not make sense, as certain references need to stay []
        # (e.g. inputs, but probably also others)
        super(Merge, self).__init__(name=name, track=track, **merge)
        # for convenience copy the dict entries too:
        for s in subgraphs:
            if hasattr(s, '__dict__'):
                update(self.__dict__, s.__dict__, overwrite=False)

    def _gen_input_vars(self):
        for outer in self.copied_subgraphs:
            for iref in set(outer).intersection(inputting_references):
                if isinstance(outer[iref], Sequence):
                    idx = 0
                    while idx < len(outer[iref]):
                        ivar = outer[iref][idx]
                        matched = yield ivar, outer
                        if matched:
                            del outer[iref][idx]
                            # keep same idx, as current idx was deleted
                        else:
                            idx += 1
                else:
                    matched = yield outer[iref], outer
                    if matched:
                        del outer[iref]

    def _delete_corresponding_output(self, ivar, outer):
        """ returns whether a matching output was found and deleted """
        for inner in self.copied_subgraphs:
            if inner == outer:
                continue  # don't look within the same graph
            for oref in set(inner).intersection(outputting_references):
                if isinstance(inner[oref], list):
                    with ignored(ValueError):  # throws ValueError if ivar not in list
                        inner[oref].remove(ivar)
                        return True
                else:
                    if inner[oref] == ivar:
                        del inner[oref]
                        return True
        return False
