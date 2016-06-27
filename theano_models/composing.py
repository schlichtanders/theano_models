#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

from itertools import izip

import theano
from schlichtanders.mycontextmanagers import ignored
from schlichtanders.myfunctools import convert
from schlichtanders.mylists import return_list, as_list
from theano import gof
import theano.tensor as T
from base import Model, Merge, track_merge, models_as_outputs, inputting_references, outputting_references
from collections import Sequence

from base_tools import total_size
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

    @models_as_outputs
    @track_merge(merge, ignore_references=outputting_references, extra_inputs=[invertible_model['norm_det']])
    def logP(y):
        invertible_model.inverse['inputs'] = y
        return base_prob_model['logP'](invertible_model.inverse['outputs']) - T.log(abs(invertible_model['norm_det']))  # equation (5)

    # adapt invertible_model too, as otherwise Y['invertible_modellogP'] would not mirror the sampler invertible_model['outputs']
    invertible_model['logP'] = logP
    merge['logP'] = logP
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


def variational_bayes(Y, randomize_key, Xs, priors=None, kl_prior=None, merge_priors=True):
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

    merge_priors : bool
        if True, the prior is also merged into the final model
        this is usually not wanted, as there are reasons that the prior parameters are learned via Hyperparametersearch.
    """
    if randomize_key not in inputting_references:
        raise ValueError("Only an inputting reference makes sense for `randomize_key`.")
    if kl_prior is None and priors is None:
        raise ValueError("Either prior or kl_prior must be given")
    if 'loglikelihood' in Y:
        # variational lower bound detected
        raise RuntimeError("Cannot perform variational lower bound twice on the same Y.")
        # reason: we don't want/cannot redo the proxifying, hence when executing variational_bayes a second time on the
        # same randomize_key, the OLD Xs get substituted, which probably aren't part of the second call at all.
        # this side effect cannot be intended

    # Preprocess args
    # ---------------
    Xs = convert(Xs, list)
    priors = convert(priors, list)

    # if kl is not given we default to the approximation of the Variational Lower Bound found in equation (2) of the
    # paper "Blundell et al. (2015) Weight uncertainty in neural networks"
    # [if we would be able to do math on Theano variables (or use sympy variables for everything)
    # we could also compute the Kullback-Leibler divergence symbolically, however this is not (yet?) the case]
    if kl_prior is None:
        # as we assume independent sub distribution,
        # the overall log_prior_distr or log_posterior_distr can be computed as a sum of the single log
        # Note, that the variational lower bound requires to substitute RV from X into every distribution function
        def log_prior_distr():
            for prior, X in izip(priors, Xs):
                if isinstance(prior, Model):
                    prior = prior['logP']  # merge must be done outside
                # else prior is already a logP function
                yield prior(X)

        def log_posterior_distr():
            for X in Xs:
                yield X['logP'](X)

        Y['logposterior'] = T.add(*log_posterior_distr())
        Y['logprior'] = T.add(*log_prior_distr())
        kl_prior = Y['logposterior'] - Y['logprior']
        # kl_prior.name = "kl_prior"  # automatically given
        extra_inputs = shallowflatten_keep_vars(x['outputs'] for x in Xs)
    else:
        extra_inputs = [kl_prior]

    # core variational bayes
    # ----------------------
    Y['kl_prior'] = kl_prior

    Y['loglikelihood'] = Y['logP']
    if "n_data" not in Y:
        Y['n_data'] = theano.shared(1, "n_data")  # needs to be updated externally, therefore real theano.shared variable here

    Y[randomize_key] = Xs  # make original parameters random

    subgraphs = [Y]
    subgraphs += Xs
    subgraphs += priors

    merge = Merge(*(sg for sg in subgraphs if isinstance(sg, Model)),
                  name="variational_lower_bound")

    # the variational lower bound as approximation of logP:
    @models_as_outputs
    @track_merge(merge, ignore_references=outputting_references, extra_inputs=extra_inputs)
    def logP(rv):
        return Y['loglikelihood'](rv) - T.inv(Y['n_data']) * kl_prior

    # adapt Y as well, as otherwise Y['logP'] would not mirror the sampler Y['outputs']
    Y['logP'] = logP
    merge['logP'] = logP
    return merge



"""
adaptations of graphs
=====================
"""

def fix_params(model):
    fix = {p: None for p in inputting_references if "parameter" in p}
    return Merge(model, name=model.name+"_fixed", **fix)
