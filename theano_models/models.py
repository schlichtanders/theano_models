#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

import theano.tensor as T
import theano
from base import Model
from schlichtanders.mydicts import IdentityDict
from copy import copy, deepcopy
from itertools import izip
from collections import Mapping
from theano import gof


"""
Deterministic Modelling
=======================

Standard Graphs are deterministic in the sense that they represent functions inputs -> outputs.

Standard Deterministic Model
----------------------------

In combination with referenced parameters, which are optimizable, we in fact already get a well defined
deterministically optimizable model
"""


def deterministic_optimizer_premap(distance=None):
    """ builds premap for a standard deterministic model

    Parameters
    ----------
    distance : metric, function working on two lists of theano expressions
        comparing targets (given as extra input for optimizer) with outputs
        Defaults to standard square loss.

    Returns
    -------
    standard premap for optimizer
    """
    if distance is None:
        def distance(targets, outputs):
            """ targets and outputs are assumed to be *lists* of theano variables """
            summed_up = 0
            n = 0
            for t, o in izip(targets, outputs):
                s = (t - o)**2
                n += s.size
                summed_up += s.sum()
            return summed_up / n

    def premap(graph):
        if isinstance(graph['outputs'], gof.graph.Variable):
            targets = [graph['outputs'].type()]
            outputs = [graph['outputs']]
        else:
            targets = [o.type() for o in graph['outputs']]
            outputs = graph['outputs']

        return IdentityDict(
            lambda key: graph[key],
            loss_inputs= targets + graph['inputs'],
            loss= distance(targets, outputs)
        )
    return premap


class DeterministicModel(Model):
    """ models prediction of some y (outputs) given some optional xs (inputs)

    hence, loss compares outputs with targets, while outputs depend on some input

    Subclassing Constraints
    -----------------------
    As the deterministic model only defines a OptimizableGraph interface, subclasses might want to additionally
    offer the extended AnnealingOptimizableGraph interface of 'loss_data' and 'loss_regularizer'.

    In order to easily interact with the initilization of a DeterministicModel in a correct way
    those two parameters should be linked in the Graph itself. Note that in this setting 'loss_inputs' and 'loss'
    become remapped as in a standard deterministic model, which is usually intended.

    If a complete new interface is wanted, please overwrite __optimizer_premap__ after creating the instance
    """

    def __init__(self, outputs, parameters, inputs=None, distance=None, **further_references):
        """ constructs general deterministic model

        Parameters
        ----------
        parameters : list of theano expressions
            parameters to be optimized
        outputs : list of theano operator or Graph
            outputs of the model, depending on inputs
        inputs : list of theano expressions
            as usual (e.g. data types which the model can use for prediction)
        distance : metric, function working on two lists of theano expressions
            comparing targets (given as extra input for optimizer) with outputs
            Defaults to standard square loss.
        further_references : kwargs
            other references
        """
        if inputs is None:
            inputs = []

        super(DeterministicModel, self).__init__(
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            **further_references
        )
        # could in principal be called before the constructor, however this order seems to make sense for a postmap:
        self.add_postmap(deterministic_optimizer_premap(distance))


#: alias
FunctionApproximator = DeterministicModel


"""
Probabilistic Modelling
=======================

A probabilistic model looks very similar to a deterministic one. The only difference is that instead of having
deterministic variables as output, a probabilistic model returns random variables.

Random variables in this sense are Theanos shared random variables (which links a numpy random number generator
internally). This means, that random variables can be used almost exactly like determinstic variables (they even have
the same theano type). The crucial different is that when called(/executed) again and again, a random variable will
return different outputs. (think of a random variables as a sampler essentially).

For doing all the math part correctly, it is however not enough to have a sampler as a random variable (RV), one also
needs information about the distribution from which the RV samples. It turns out, that it is not trivial to express
the probability distribution in theano. We want a probability distribution to be
    1. a theano expression
    2. automatically differentiable (Theano grad)
    3. automatically substitutable (Graph)
    4. be transformable into a function, returning the density/probability value for a certain random variable
The crucial point is, how to denote the random variable within the probability distribution to ensure the four points
above.
The naive guess, that one could simply use the sampler random variable within the probability distribution (P)
revealed to kick (2) (e.g. RV = mean + noise, P = abs(RV - mean).sum() makes it clear, that P is not differentiable
with respect to mean if RV is used directly).
Using a function rv -> P(rv)  kicks (1) and (3).
Using a tuple/list [rv, P] kicks (1) and (3) (I think (3) could even work given the current rigorous implementation of
substitute, looks however a bit more like magic)

The solution we came up with is twofold:
1.  In a ProbabilisticModel itself, the rv of the probability distribution P gets a extra reference called "P_RV"
    and can be cleanly substituted like any other graph reference by ``graph['P_RV'] = replacement``.
    "P" refers to the complete scalar theano expression.
    "RV" and "outputs" refer both to the sampler.

2.  In addition, as an function argument, you can also pass a function P: rv -> P(rv) giving the distribution function.
    Of course, this makes only sense if no full ProbabilisticModel is needed (i.e. no sampler, but only distribution
    function, for instance). This revealed to be intuitive and straightforward.

Note, that theano makes it easy to optimize with this kind of structure. A random variable 'RV' can itself be a more
complex theano expression, for which e.g. shared parameter gradients can be automatically inferred by theano.
Updating these shared parameters with an optimizer automatically updates also the distribution 'P', as it must depend
on the very same parameters.


Standard Probabilistic Model
----------------------------
"""


def probabilistic_optimizer_premap(graph):
    # virtual random variable
    # (we cannot use graph['RV'] itself, as the automatic gradients will get confused because graph['RV'] is a complex sampler)
    RV = graph['RV'].type()  # like targets for deterministic model
    return IdentityDict(
        lambda key: graph[key],
        loss_inputs = [RV] + graph['inputs'],
        loss = -graph['logP'](RV)
    )


class ProbabilisticModel(Model):
    """ Returns a random variable with Probability Distribution attached (.P)

    While the random variable describes some target, its distribution can depend on further parameters. This can be
    used for noise modeling ("unsupervised"), prediction ("supervised"), ...

    The optimizer optimizes probability of Data = (Targets, ExtraInformations), which is interpreted as probability
    of the Targets given the optional extra information.

    Probabilistic models are relatively intuitive to use, however they have a rather intricate internal
    building (which is due to how probability works).

    Concrete, for a conditional probabilistic model inputs work like further parameters, and the outputs is the
    target random variable (with respective distribution function).

    Note
    ----
    A probabilistic model is meant to have only one output. Several outputs could be interpreted as
    *independent* random variables, however then they should not depend on a same set of expressions / parameters and
    hence should be separated in distinct probabilistic models.

    Comparison to DeterministicModel
    --------------------------------
    A deterministic model outputs a kind of deterministic prediction.
    In a respective probabilistic model, the output is now a *random* variable y, i.e. instead of a prediction,
    a distribution over predictions is returned.
    Both models can depend on some extra inputs (deterministic in both models).


    Subclassing Constraints
    -----------------------
    As the probabilistic model only defines a OptimizableGraph interface, subclasses might want to additionally
    offer the extended AnnealingOptimizableGraph interface of 'loss_data' and 'loss_regularizer'.

    In order to easily interact with the initilization of a ProbabilisticModel in a correct way
    those two parameters should be linked in Graph itself. Note that in this setting 'loss_inputs' and 'loss' become
    remapped as in a standard probabilistic model.

    If a complete new interface is wanted, please overwrite __optimizer_premap__ after creating the instance
    """

    def __init__(self, RV, logP, parameters, inputs=None, **further_references):
        """
        Parameters
        ----------
        RV: random variable, theano expression (shared random stream)
            output random variable which probability distribution shall be optimized.
            The concrete sampler can depend on the inputs.
            Note: if you want to have several outputs, build a custom multi-dimensional RV, so that the product
            probability distribution is clear. (may support standard iid list in the future)

        logP : function RV -> scalar theano expression
            the log probability distribution which depends on the function argument, if P is function.
            The probability distribution can further depend on the input.

        parameters : list of theano expressions
            parameters to be optimized eventually

        inputs : list of theano expressions
            extra information needed to compute the probability distribution of RV
        """
        if inputs is None:
            inputs = []

        super(ProbabilisticModel, self).__init__(
            inputs=inputs,
            outputs=RV,
            RV=RV,  # direct alias
            logP=logP,
            parameters=parameters,
            **further_references
        )
        # could in principal be called before the constructor, however this order seems to make sense for a postmap:
        self.add_postmap(probabilistic_optimizer_premap)


#: alias
DistributionApproximator = ProbabilisticModel


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


def variational_bayes(Y, randomize_key, Xs, priors=None, kl_prior=None, merge_keys=('parameters', 'inputs'), clone_prior=True):
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

    kl_prior : theano expression or Graph (if parameters need to be merged)
        kullback leibler divergence KL(X.P||prior). This does not depend any long on X if I understood it correctly,
        but only on hyperparameters.

    merge_keys : list of str
        keys to be merged from all Graph objects part of this operation. Defaults to ['parameters', 'inputs']
    """
    if kl_prior is None and priors is None:
        raise ValueError("Either prior or kl_prior must be given")

    # Preprocess args
    # ---------------

    # if kl is not given we default to the approximation of the Variational Lower Bound found in equation (2) of the
    # paper "Blundell et al. (2015) Weight uncertainty in neural networks"
    # [if we would be able to do math on Theano variables (or use sympy variables for everything)
    # we could also compute the Kullback-Leibler divergence symbolically, however this is not (yet?) the case]
    if kl_prior is not None:
        # no substitution necessary as kl_prior should not depend on X anylonger, but only on hyperparameters.
        if isinstance(kl_prior, Model):
            kl_prior = kl_prior['outputs']  # assumes single output
        # else
        # kl_prior is already standard expression
    else:
        # as we assume independent sub distribution,
        # the overall log_prior_distr or log_posterior_distr can be computed as a sum of the single log
        # Note, that the variational lower bound requires to substitute RV from X into every distribution function
        log_prior_distr = 0.0
        for prior, X in zip(priors, Xs):
            if isinstance(prior, ProbabilisticModel):
                prior = prior['logP']  # merge must be done outside
            # else prior is already a logP function
            log_prior_distr += prior(X['RV'])

        log_posterior_distr = 0.0
        for X in Xs:
            log_posterior_distr += X['logP'](X['RV'])

        kl_prior = log_posterior_distr - log_prior_distr

    # core variational bayes
    # ----------------------
    loglikelihood = Y['logP']
    if "n_data" not in Y:
        Y['n_data'] = theano.shared(1)  # needs to be updated externally

    Y[randomize_key] = Xs  # make original parameters random
    def variational_lower_bound(RV):
        return loglikelihood(RV) - 1/Y['n_data'] * kl_prior
    Y['logP'] = variational_lower_bound  # functions do not get proxified, so this is not a loop