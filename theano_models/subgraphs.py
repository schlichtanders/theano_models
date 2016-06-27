#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import warnings
from collections import Sequence, MutableMapping, Mapping, defaultdict
from copy import copy
from itertools import izip
from pprint import pformat
import wrapt
from functools import wraps
import theano.tensor as T
from schlichtanders.mycontextmanagers import until_stopped, ignored
from schlichtanders.mydicts import update
from schlichtanders.myfunctools import fmap, convert
from schlichtanders.mylists import remove_duplicates, shallowflatten

from util import clone, as_tensor_variable, deepflatten_keep_vars, U, shallowflatten_keep_vars
from util.theano_helpers import is_clonable, get_graph_inputs, is_pseudo_constant
import types

import theano

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


