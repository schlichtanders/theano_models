#!/usr/bin/python
# -*- coding: utf-8 -*-

# flat core
from base import Model, Merge, track_model, track_merge, Reparameterize, Flatten
from base import inputting_references, outputting_references, models_as_outputs, model_to_output
from placeholders import Placeholder
from composing import variational_bayes, normalizing_flow, fix_params

# sub packages:
import tools
import postmaps
import probabilistic_models
import deterministic_models
import data

# important important class methods for direct access:
check_all_identities = deterministic_models.InvertibleModel.check_all_identities
reduce_all_identities = check_all_identities  # almost DEPRECATED

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'
