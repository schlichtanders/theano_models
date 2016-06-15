#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import izip
import theano.tensor as T
from theano_helpers import as_tensor_variable, clone, is_clonable, clone_all, PooledRandomStreams # because this is so central
from theano import gof
from collections import Sequence, MutableMapping
from schlichtanders.mymeta import proxify
from schlichtanders.myfunctools import fmap
from schlichtanders.mylists import remove_duplicates, return_list
from schlichtanders.mydicts import update, DefaultDict, ModifyDict
import itertools as it
from collections import defaultdict
import types
from copy import copy

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


@return_list
def shallowflatten_keep_vars(th):
    if isinstance(th, gof.Variable):  # need to be first, as gof.Variables are also Sequences
        yield th
    elif hasattr(th, '__iter__'):
        for t in th:
            if isinstance(t, gof.Variable):
                yield t


def deepflatten_keep_vars(sequences_of_theano_variables):
    return list(deepflatten_keep_vars_(sequences_of_theano_variables))


def deepflatten_keep_vars_(th):
    if isinstance(th, gof.Variable):  # need to be first, as gof.Variables are also Sequences
        yield th
    elif isinstance(th, Sequence) or isinstance(th, types.GeneratorType):  # could this be more general?
        for t in th:
            for i in deepflatten_keep_vars_(t):
                yield i


# starting counting at 1 seems much more intuitive here
_start_idx = 1
_running_numbers = defaultdict(lambda: it.count(_start_idx))

def get_unique_name(name, ommit_first_index=True):
    """ appends running number to name to make it unique """
    count = _running_numbers[name]
    idx = next(count)
    if ommit_first_index and idx == _start_idx:
        return name
    return "%s%i" % (name, idx)

#: convenience shortcut
U = get_unique_name