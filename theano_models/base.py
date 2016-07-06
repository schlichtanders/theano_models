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
import theano
import theano.tensor as T
from theano.gof.fg import MissingInputError
from schlichtanders.mycontextmanagers import until_stopped, ignored
from theano import gof, config
from theano.compile.sharedvalue import SharedVariable

from schlichtanders.mydicts import update, ModifyDict
from schlichtanders.mylists import sequencefy, remove_duplicates, as_list
from schlichtanders.mymeta import proxify
from schlichtanders.myfunctools import fmap, convert

from util import clone, as_tensor_variable, shallowflatten_keep_vars, deepflatten_keep_vars, U, reset_eval
from util.theano_helpers import is_clonable, get_graph_inputs, clone_all, is_pseudo_constant
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
                        # elif unique_name:
                        #     x.name = U(x.name)
                        # not useful, as this also applies to variables of already declared models which are reused here...
            else:
                if hasattr(v, 'name'):
                    if v.name is None:
                        v.name = "%s.%s" % (self.name, k)
                    # elif unique_name:
                        # v.name = U(v.name)

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
        def len_min_sum(iterable):
            return len(iterable) - sum(iterable)
        def new_reshaped():
            for o, n in izip(old, new):
                if (o.broadcastable != n.broadcastable
                        and len_min_sum(o.broadcastable) == len_min_sum(n.broadcastable)):  # counts False
                    idx = itertools.count()
                    def broadcast_pattern():
                        for b in o.broadcastable:
                            if b:
                                yield 'x'
                            else:
                                yield next(idx)
                    yield n.squeeze().dimshuffle(*broadcast_pattern())
                else:
                    yield n

        assert all(o.type == n.type for o, n in izip(old, new_reshaped())), "No substitution as theano types differ"
        for o, n in izip(old, new):
            proxify(o, n)
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
        subgraphs : Model or dict of references
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
        self.copied_subgraphs = map(copy, subgraphs)

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


"""
Core Models
===========
Models which are used everywhere and always, hence belong to the core. At the moment, these encompass mainly
models used for reparameterization purposes.
"""


class Reparameterize(Model):
    """ This class is for a clean transformation of parameters which interacts well with Merge and visualization

    In General what is done is that for each param in parameters::
        new_param = finv(param)
            param = f(new_param)

    The underlying new_params are listed as parameters in the model, while the reparameterized params are outputs
    """
    # TODO raise error/warning if parameters are already proxified
    def __init__(self, parameters, f, finv, givens={}):
        """
        Parameters
        ----------
        parameters : list of theano variables
            to be reparameterized
        f : function theano_variable -> theano_variable
        finv : function theano_variable -> theano_variable
        """
        is_singleton = not isinstance(parameters, Sequence)
        parameters = convert(parameters, Sequence)
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
            proxify(p, new_p)
            underlying_parameters.append(underlying_p)
        # return new_underlying_parameters[0] if is_singleton else new_underlying_parameters
        super(Reparameterize, self).__init__(
            inputs=[],
            outputs=parameters[0] if is_singleton else parameters,
            parameters=underlying_parameters
        )


class Center(Model):
    def __init__(self, parameters):
        parameters = convert(parameters, Sequence)
        try:
            copies = theano.function([], parameters)()
        except MissingInputError as e:
            warnings.warn("MissingInputs. Using symbolic version, might be considerably slower. %s" % e)
            assert all(is_clonable(p) for p in parameters), "can only center clonable parameters"
            copies = [clone(p) for p in parameters]

        # this works for both numeric or symbolic "copies"
        zeros = [T.zeros(cp.shape) for cp in copies]
        for z, p in izip(zeros, parameters):
            z.name = str(p) + "_centered"
        for p, z, cp in izip(parameters, zeros, copies):
            new_name = str(p) + "_centered"
            proxify(p, z + cp)
            p.name = new_name

        super(Center, self).__init__(
            inputs=[],
            parameters=zeros,
            outputs=parameters
        )





class Flatten(Model):
    def __init__(self, parameters, flat_key="flat", givens={}):
        """ this does not work with subgraphs as substitution is required

        Parameters
        ----------
        model
        key
        initial_inputs
        """
        # TODO raise error/warning if parameters are already proxified
        try:
            if not isinstance(parameters, Sequence):
                raise ValueError("`parameters` is not Sequence. Nothing to flat.")
            flat_sym = T.concatenate([p.flatten() for p in parameters])
            shapes_sym = [p.shape for p in parameters]
            _f = theano.function([], [flat_sym] + shapes_sym, on_unused_input="warn", givens=givens, mode="FAST_COMPILE")()
            flat_num, shapes_num = _f[0], _f[1:]
            flat = as_tensor_variable(flat_num)
            # we need extra escaping that this works with d3viz and graphviz, because colon : in names has extra semantics
            # see http://stackoverflow.com/questions/31523810/pydot-error-involving-parsing-character-followed-by-number
            flat.name = '"%s"' % ':'.join((p.name or str(p)) for p in parameters)

            i = 0
            for p, shape in izip(parameters, shapes_num):
                # for size and shapes we need to refer to the copies, as the original parameters get proxified
                # (and size/shape refer to the parameters again)
                size = np.prod(shape)
                new_p = flat[i:i + size].reshape(shape)
                new_p.name = (p.name or str(p)) + "_flat"
                proxify(p, new_p)
                i += size

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
            flat.name = '"%s"' % ":".join(cp.name for cp in copies)
            # Subgraph({'inputs': [], 'outputs': flat}, "symbolic_flat")  # to encapsulate the mess with clone_all TODO needed? seemed buggy

            i = 0
            for p, cp in izip(parameters, copies):
                # for size and shapes we need to refer to the copies, as the original parameters get proxified
                # (and size/shape refer to the parameters again)
                new_p = flat[i:i + cp.size].reshape(cp.shape)
                new_p.name = (p.name or str(p)) + "_flat"   # unnecessary if FlatKey is its own Model
                proxify(p, new_p)
                i += cp.size

        super(Flatten, self).__init__(**{
            'inputs': [],  # if flat_key='inputs' this gets overwritten by default dict behaviour
            flat_key: flat,
            'outputs': parameters,
        })
