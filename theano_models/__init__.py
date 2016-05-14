#!/usr/bin/python
# -*- coding: utf-8 -*-

from base import Model, reset_eval, Merge, merge_key, merge_inputs
from placeholders import Placeholder

# standard helpers
from util import as_tensor_variable, clone, is_clonable
# standard norms and distances
from util import norm_distance, L1, L2
# reparameterization
from util import total_size, reparameterize_map, softplus, softplus_inv, squareplus, squareplus_inv

import postmaps
import probabilistic_models
import deterministic_models
import data

# important class methods:
from deterministic_models import InvertibleModel

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'
