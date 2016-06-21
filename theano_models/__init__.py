#!/usr/bin/python
# -*- coding: utf-8 -*-

# core
from base import Model, reset_eval
from subgraphs import inputting_references, outputting_references
from subgraphs import Subgraph, subgraphs_as_outputs, subgraph_to_output, subgraph, subgraph_extra, subgraph_modify
from base import Merge, Reparameterize, Flatten, merge_key #, merge_inputs, merge_key_reparam, pmerge_key_reparam
from placeholders import Placeholder

# standard helpers
# util is now included in tools for better user convenience
# from util import as_tensor_variable, clone, is_clonable, clone_all, PooledRandomStreams, get_inputs, get_profile, clone_renew_rng
import tools

from composing import variational_bayes, normalizing_flow

import postmaps
import probabilistic_models
import deterministic_models
import data

# important important class methods for direct access:
reduce_all_identities = deterministic_models.InvertibleModel.reduce_all_identities

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'
