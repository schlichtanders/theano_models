#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import izip
import theano.tensor as T
from theano_helpers import as_tensor_variable, clone, is_clonable, clone_all  # because this is so central
from theano import gof
from collections import Sequence
from schlichtanders.mymeta import proxify
from schlichtanders.myfunctools import fmap
from schlichtanders.mylists import remove_duplicates
from schlichtanders.mydicts import update
import types

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


def deepflatten_keep_vars(sequences_of_theano_variables):
    return list(deepflatten_keep_vars_(sequences_of_theano_variables))


def deepflatten_keep_vars_(th):
    if isinstance(th, gof.Variable):  # need to be first, as gof.Variables are also Sequences
        yield th
    elif isinstance(th, Sequence) or isinstance(th, types.GeneratorType):  # could this be more general?
        for t in th:
            for i in deepflatten_keep_vars_(t):
                yield i