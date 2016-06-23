#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import warnings
from collections import Sequence, MutableMapping, Mapping
from copy import copy
from itertools import izip
from pprint import pformat
import wrapt
from functools import partial

import numpy as np
import theano
import theano.tensor as T
from theano.gof.fg import MissingInputError
from schlichtanders.mycontextmanagers import until_stopped, ignored
from theano import gof, config
from theano.compile.sharedvalue import SharedVariable

from schlichtanders.mydicts import update, ModifyDict
from schlichtanders.mylists import sequencefy, remove_duplicates
from schlichtanders.mymeta import proxify
from schlichtanders.myfunctools import fmap, convert

from util import clone, as_tensor_variable, deepflatten_keep_vars, U
from util.theano_helpers import is_clonable, get_inputs, clone_all
import types

from subgraphs import subgraph_to_output, subgraphs_as_outputs, Subgraph, inputting_references, outputting_references
from subgraphs_tools import complex_reshape

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

inputting_references.add("flat")

theano.function
theano.compile.pfunc
theano.gof.opt.SeqOptimizer
theano.gof.opt.MergeOptimizer
theano.tensor.log
theano.gof.fg
from theano.tensor.shared_randomstreams import RandomStreams

"""
Subgraph with substitution == Model
===================================
This is the core class of the whole module. It simply overwrites dict assigment with a substitution mechanism to
enable interactive chaining of models.
"""


class Model(Subgraph):
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

    @subgraphs_as_outputs
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
        further_references: kwargs of theano expressions, or lists thereof
            possible further references
        """
        if inputs is None:
            inputs = get_inputs(outputs)
        inputs = convert(inputs, Sequence)

        super(Model, self).__init__(name=name,
            outputs=outputs,
            inputs=inputs,
            **further_references
        )

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
        old = convert(old, Sequence)

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
            print("fancy reshaping")
            old_cp = [clone(o) for o in
                      old]  # we need copy as new gets proxified later on, single copy satifies as this is not recursive
            new = complex_reshape(new, old_cp)  # list(...) not needed because of @track_as_helper decorator
        else:
            singleton = True
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
            self.references[key] = new[0] if singleton else new
            return

        # substitution of theano variables
        # ================================
        assert all(o.type == n.type for o, n in izip(old, new)), "No substitution as theano types differ"
        for o, n in izip(old, new):
            proxify(o, n)
        # make sure that simply all cached compiled functions get destroyed, as references are no longer valid
        reset_eval(self)

    @subgraphs_as_outputs
    def __setitem__(self, key, value):
        """ convenience access to substitute_key """
        if key in self:
            self.substitute_key(key, value)
        else:
            if not isinstance(value, self.ALLOWED_VALUETYPES):
                raise TypeError(
                    "The type of the given value is not supported. You may change ``Model.ALLOWED_VALUETYPES`` if you know what your doing.")
            self.references[key] = value

    def add_new(self, key, value):
        """ for writing self-explanatory code """
        if key in self:
            raise ValueError("Key %s is already part of the model." % key)
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


class Merge(Model):
    """ this class is used for combining references of several models in a consistent manner """

    def __init__(self, *subgraphs, **other_references):
        """
        Parameters
        ----------
        subgraphs : Subgraph
            to be combined consistently
        name : str
            like in Model
        convert_singletons_to_list : Bool, default False
            if True then all single variables are wrapped into lists if combined with other subgraphs
            if False then the first reference is taken
        other_references : None, str, gof.Variables, list of gof.Variables
            if is None, then the respective key is deleted and replaced by empty list []
            if keywordarg="str" then the reference is understood as a renaming  new_key=old_key
            other references are simply combined with old references
        """
        convert_singletons_to_list = other_references.pop("convert_singletons_to_list", False)
        name = other_references.pop("name", None)
        self.original_subgraphs = subgraphs
        self.copied_subgraphs = map(copy, subgraphs)

        # remove all nested references:
        input_vars = self._gen_input_vars()
        with until_stopped:
            ivar = input_vars.next()
            while True:
                matched = self._delete_corresponding_output(ivar)
                ivar = input_vars.send(matched)

        # merge subgraphs:
        merge = {}
        subgraph_references = set(self.copied_subgraphs[0]).union(*self.copied_subgraphs[1:])

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
                del merge[key]
                merge[key] = []

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
                if key not in merge:
                    merge[key] = other_references[key]
                elif isinstance(merge[key], list) or convert_singletons_to_list:
                    merge[key] = convert(merge[key], list) + convert(other_references[key], list)
                # else nothing happens as subgraphs come before other_references (by syntax)

        # deleting empty lists at the end does not make sense, as certain references need to stay []
        # (e.g. inputs, but probably also others)
        super(Merge, self).__init__(name=name, ignore=True, **merge)
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
                        matched = yield ivar
                        if matched:
                            del outer[iref][idx]
                            # keep same idx, as current idx was deleted
                        else:
                            idx += 1
                else:
                    matched = yield outer[iref]
                    if matched:
                        del outer[iref]

    def _delete_corresponding_output(self, ivar):
        """ returns whether a matching output was found and deleted """
        for inner in self.copied_subgraphs:
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
            underlying_p = subgraph_to_output(finv(cp))
            underlying_p.name = p.name + "_" + f.func_name  # naming is not needed if f, finv are Models
            new_p = subgraph_to_output(f(underlying_p))
            new_p.name = (p.name or str(p)) + "_reparam"
            proxify(p, new_p)
            underlying_parameters.append(underlying_p)
        # return new_underlying_parameters[0] if is_singleton else new_underlying_parameters
        super(Reparameterize, self).__init__(
            inputs=[],
            outputs=parameters[0] if is_singleton else parameters,
            parameters=underlying_parameters
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


def merge_key(models, key="parameters"):
    """ simply combines all model[key] values for model in models """
    parameters = []
    for g in models:
        if key in g:
            parameters += g[key]
    return parameters