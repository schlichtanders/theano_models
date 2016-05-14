#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import theano
import theano.tensor as T
from base import Model, models_as_outputs
__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


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
    invertible_model(base_prob_model)  # output not needed, but assigns input
    @models_as_outputs
    def normalized_flow(y):
        # equation (5)
        return base_prob_model['logP'](invertible_model.inv(y)) - T.log(abs(invertible_model['norm_det']))

    invertible_model['logP'] = normalized_flow
    return invertible_model


"""
Composing Probabilistic Models
------------------------------

One famous framework for composing probabilistic models is bayesian modelling.

In a bayesian setting the distribution function RV.P is often an Expectation over the distribution of the weights W
used for modelling RV. In formulas:
    ``RV.P = E(P(RV | W), W.P)`` where ``P(RV | W)``, the 'likelihood' is known.

The problem is that the expectation is often not analytically solvable, however ``.P`` should preferably reference
a theano expression to be used within optimizers.

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
        Inputs and parameters are merged into the new parameters, and inputs of the model (in certain settings,
        these are named hyperparameters)
        (each model includes an attached probability distribution X['RV'].P)

        The list size must match Y[randomize_key] as those get randomized by these.
        Different list entries are supposed to be independent.

        The combined probability distribution is meant to be optimized to approach the true ``P(X|Data)``
        posterior distribution which gives by integration the ultimately desired distribution
        ``P(Y | Data) = E_{P(X|Data)} P(Y | X)``

    priors : list of ProbabilisticModel or functions rv -> log probability distribution
        prior probability distribution of the weights (lists must match Ws and Y[randomize_key])
        Different list entries are supposed to be independent.

    kl_prior : theano expression or Model (if parameters need to be merged)
        kullback leibler divergence KL(X.P||prior). This does not depend any long on X if I understood it correctly,
        but only on hyperparameters.
    """
    if kl_prior is None and priors is None:
        raise ValueError("Either prior or kl_prior must be given")

    # Preprocess args
    # ---------------

    # if kl is not given we default to the approximation of the Variational Lower Bound found in equation (2) of the
    # paper "Blundell et al. (2015) Weight uncertainty in neural networks"
    # [if we would be able to do math on Theano variables (or use sympy variables for everything)
    # we could also compute the Kullback-Leibler divergence symbolically, however this is not (yet?) the case]
    if kl_prior is None:
        # as we assume independent sub distribution,
        # the overall log_prior_distr or log_posterior_distr can be computed as a sum of the single log
        # Note, that the variational lower bound requires to substitute RV from X into every distribution function
        log_prior_distr = 0.0
        for prior, X in zip(priors, Xs):
            if isinstance(prior, Model):
                prior = prior['logP']  # merge must be done outside
            # else prior is already a logP function
            log_prior_distr += prior(X)

        log_posterior_distr = 0.0
        for X in Xs:
            log_posterior_distr += X['logP'](X)

        kl_prior = log_posterior_distr - log_prior_distr

    # core variational bayes
    # ----------------------

    Y['kl_prior'] = kl_prior

    Y['loglikelihood'] = Y['logP']
    if "n_data" not in Y:
        Y['n_data'] = theano.shared(1)  # needs to be updated externally, therefore real theano.shared variable here

    Y[randomize_key] = Xs  # e.g. make original parameters random

    @models_as_outputs
    def variational_lower_bound(rv):
        return Y['loglikelihood'](rv) - 1 / Y['n_data'] * kl_prior
    Y['logP'] = variational_lower_bound  # functions do not get proxified, so this is not a loop
    return Y

