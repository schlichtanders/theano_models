#!/usr/bin/python
# -*- coding: utf-8 -*-

# flat core
from base import (Model, Merge,
                  inputting_references, outputting_references, models_as_outputs, model_to_output)
from loss import (loss_deterministic, loss_probabilistic, loss_regularizer, loss_variational, loss_normalizingflow,
                  numericalize, scipy_kwargs, climin_kwargs, AnnealingCombiner)
from base_tools import(
    as_merge, track_merge, as_model, track_model, as_proxmodel, track_proxmodel,
    fct_to_inputs_outputs, get_equiv_by_name,
    total_size, complex_reshape,
    L1, L2, norm_distance,
    softmax, softmax_inv,
    tan_01_R, tan_01_R_inv, square_01_R, square_01_R_inv, logit, logistic,
    softplus, softplus_inv, squareplus, squareplus_inv,
    prox_center, prox_flatten, prox_reparameterize
)
from util import *

from placeholders import Placeholder
from composing import variational_bayes, normalizing_flow, fix_params
from visualization import d3viz

# sub packages:  # TODO need to be imported here?
import probabilistic_models
import deterministic_models
import data

# important important class methods for direct access:
check_all_identities = deterministic_models.InvertibleModel.check_all_identities
reduce_all_identities = check_all_identities  # almost DEPRECATED

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'
# TODO for documentation using sphinx ``__all__ = ...`` is needed


T.shape