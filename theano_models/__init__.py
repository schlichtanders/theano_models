#!/usr/bin/python
# -*- coding: utf-8 -*-

#: replaces ``shared`` in this framework:
from theano.tensor.basic import as_tensor_variable

from base import Model, reset_eval
from placeholders import Placeholder

import util
import postmaps
import probabilistic_models
import deterministic_models
import data

# important class methods:
from deterministic_models import InvertibleModel

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'
