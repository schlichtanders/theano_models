#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

from itertools import izip

import theano
from schlichtanders.mycontextmanagers import ignored
from schlichtanders.myfunctools import convert
from schlichtanders.mylists import return_list, as_list
from schlichtanders.mymeta import Proxifier
from theano import gof
import theano.tensor as T
from base import Model, Merge, models_as_outputs, inputting_references, outputting_references
from collections import Sequence

from base_tools import total_size, as_merge
from theano_models.util.theano_helpers import is_theano_proxified
from util import as_tensor_variable, shallowflatten_keep_vars
from numbers import Number

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

outputting_references.update(['loglikelihood', 'kl_prior', 'norm_dets'])
inputting_references.update(['n_data', 'extra_inputs'])

"""
Composing Deterministic and Probabilistic Models
------------------------------------------------

Note, like all wrappers, also these are composable with standard compose function.
"""


# this is not meant as a class, but as a composer, which returns the respective Merge
def normalizing_flow(invertible_model, base_prob_model):
    """ transforms ``invertible_model['inputs'] = base_prob_model`` while adding correct ``logP``

    No merging is performed. Do this separately.

    Note
    ----
    In case you wanna use self[P_RV] = self[RV] further simplifications apply due to the invertible models.
    Simply run InvertibleModel.reduce_all_identities()

    Parameters
    ----------
    base_prob_model : ProbabilisticModel
        defines z
    invertible_model : InvertibleModel
        invertible_model(base_base_prob_model) is new probabilistic model, i.e. it transforms the base_prob_model
    """
    invertible_model['inputs'] = base_prob_model
    merge = Merge(invertible_model, base_prob_model, name="normalized_flow")

    # logP
    # ----
    rv = merge['outputs'].type("rv")
    invertible_model.inverse['inputs'] = rv
    base_prob_model.logP['inputs'] = invertible_model.inverse['outputs']

    logP = Merge(invertible_model.inverse, base_prob_model.logP, name='normalized_flow.logP', track=True,
        extra_inputs=[invertible_model['norm_det']],
        outputs=base_prob_model.logP['outputs'] - T.log(abs(invertible_model['norm_det']))
    )
    # adapt invertible_model too, as otherwise Y.logP['outputs'] would not mirror the sampler Y['outputs']
    invertible_model.logP = logP
    merge.logP = logP
    return merge


"""
Composing Probabilistic Models
------------------------------

One famous framework for composing probabilistic models is bayesian modelling.

This is wrong... bayesian setting assumes prior distribution.
[[In a bayesian setting the distribution function RV.P is often an Expectation over the distribution of the weights W
used for modelling RV. In formulas:
    ``RV.P = E(P(RV | W), W.P)`` where ``P(RV | W)``, the 'likelihood' is known.

The problem is that the expectation is often not analytically solvable, however ``.P`` should preferably reference
a theano expression to be used within optimizers.]]

There are different approaches to this problem. A relatively successful one is the so called variational lower bound
which itself is an approximation (a lower bound) of exactly this Probability distribution.
"""


def variational_bayes(Y, randomize_key, Xs, priors=None, kl_prior=None):
    """
    Models Y[randomize_key] as random variable Xs using Bayes. I.e. like Y[randomize_key] = Xs. However, the probability
    function changes, concretely Y['logP'] becomes an integral. Here, this intergral is approximated by the
    variational lower bound.

    Importantly, afterwards, the optimizer interface links to random expressions which should be averaged to get more
    accurate loss functions, gradients, etc. (see e.g. by ``schlichtanders.myoptimizers.average`` for a general optimizer
    average wrapper).

    In principal this VariationalBayes construction is recursively applicable. However note that the resulting
    distribution functions depend on more and more random variables the more layers there are. (Symbolic Integration is
    circumvented within variational methods by replacing it with random variable samplers). Hence, the distribution
    functions should be averaged to get correct results.

    IMPORTANT
    ---------
    The number of samples the algorithm is trained on must be known for the variational lower bound loss function.
    The current implementation demands that ``self['N_Data'].set_value(n)`` has to be called before optimization,
    in order to set the respective shared theano variable correctly. (defaults to 1, which usually is to low and
    hence will emphasize the kl_prior more than theoretically optimal)

    Parameters
    ----------
    Y : ProbabilisticModel
        model of a random variable, which parameters (or another key) shall be randomized by bayesian setting
        Y['RV'].P = P(Y | x) (i.e. likelihood from parameter perspectiv)
        works in place

    randomize_key : str
        denotes the reference in Y to be substituted by Xs.

    Xs : list of ProbabilisticModel
        models the Y[randomize_key] probabilistically.

        The list size must match Y[randomize_key] as those get randomized by these.
        Different list entries are supposed to be independent.

    priors : list of ProbabilisticModel or functions rv -> log probability distribution
        prior probability distribution of the weights (lists must match Ws and Y[randomize_key])
        Different list entries are supposed to be independent.

    kl_prior : theano expression or Model (if parameters need to be merged)
        kullback leibler divergence KL(X.P||prior). This does not depend any longer on X if I understood it correctly,
        but only on hyperparameters.
    """
    if randomize_key not in inputting_references:
        raise ValueError("Only an inputting reference makes sense for `randomize_key`.")
    if kl_prior is None and priors is None:
        raise ValueError("Either prior or kl_prior must be given")
    if any(is_theano_proxified(y) for y in convert(Y[randomize_key], list)):
        raise RuntimeError("Y[%s] is already proxified. For variational lower bound it is usually not intended to proxify things twice." % randomize_key)

    # Preprocess args
    # ---------------
    Xs = convert(Xs, list)
    priors = convert(priors, list)

    # if kl is not given we default to the approximation of the Variational Lower Bound found in equation (2) of the
    # paper "Blundell et al. (2015) Weight uncertainty in neural networks"
    # [if we would be able to do math on Theano variables (or use sympy variables for everything)
    # we could also compute the Kullback-Leibler divergence symbolically, however this is not (yet?) the case]
    add_refs = {}
    if kl_prior is None:
        # as we assume independent sub distribution,
        # the overall log_prior_distr or log_posterior_distr can be computed as a sum of the single log
        # Note, that the variational lower bound requires to substitute RV from X into every distribution function
        def log_prior():
            for prior, X in izip(priors, Xs):
                if hasattr(prior, 'logP'):
                    prior = prior.logP
                # else prior is already a logP Model
                prior['inputs'] = X
                yield prior['outputs']

        def log_posterior():
            for X in Xs:
                X.logP['inputs'] = X
                yield X.logP['outputs']

        add_refs['logposterior'] = T.add(*log_posterior())
        add_refs['logprior'] = T.add(*log_prior())
        kl_prior = add_refs['logposterior'] - add_refs['logprior']

    # core variational bayes
    # ----------------------
    add_refs['kl_prior'] = kl_prior

    Y.loglikelihood = Y.logP
    if "n_data" not in Y:  # this needs to be referenced in Y, as also Y's logP is updated
        Y['n_data'] = theano.shared(1, "n_data")  # needs to be updated externally, therefore real theano.shared variable here

    Y[randomize_key] = Xs  # make original parameters random

    subgraphs = [Y]
    subgraphs += Xs
    subgraphs += priors
    merge = Merge(*(sg for sg in subgraphs if isinstance(sg, Model)),
                  name="variational_lower_bound", **add_refs)

    # the variational lower bound as approximation of logP:
    logP = Merge(Y.loglikelihood, name="variational_lower_bound.logP", ignore_references=outputting_references, track=True,
                 extra_inputs=[kl_prior],
                 outputs=Y.loglikelihood['outputs'] - T.inv(Y['n_data']) * kl_prior)
    # adapt Y as well, as otherwise Y.logP['outputs'] would not mirror the sampler Y['outputs']
    Y.logP = logP
    merge.logP = logP
    return merge


"""
adaptations of graphs
=====================
"""


def fix_params(model):
    fix = {p for p in inputting_references if "parameter" in p}
    return Merge(model, name=model.name+"_fixed", ignore_references=fix)
