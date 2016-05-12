#!/usr/bin/python
# -*- coding: utf-8 -*-
from base import Model, merge, merge_parameters, merge_inputs, reparameterize_map, flatten_parameters
from util.theano_helpers import shared, softplus, softplus_inv, total_size, L2, L1, norm_distance
from placeholders import Placeholder

import postmaps
import probabilistic_models
import deterministic_models
import data

# convenience bindings to important class methods
# -----------------------------------------------
from deterministic_models import InvertibleModel as _im
reduce_all_identities = _im.reduce_all_identities

from util.theano_helpers import SymbolicSharedVariable as _ssv
evaluate_all_unevaluated = _ssv.evaluate_all_unevaluated

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'
