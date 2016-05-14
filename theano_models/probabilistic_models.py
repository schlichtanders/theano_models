#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import theano.tensor as T
import theano
from theano import config
from theano.tensor.shared_randomstreams import RandomStreams

from base import Model, models_as_outputs
from deterministic_models import InvertibleModel
from schlichtanders.mydicts import update
from util import as_tensor_variable

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


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
needs information about the distribution from which the RV samples.
We represent a distribution by a function logP: rv -> logP(rv), where logP stands for logarithmic probability. We need
logarithmic scale for numerical issues.

Note, that theano makes it easy to optimize with this kind of structure. A random variable 'RV' can itself be a more
complex theano expression, for which e.g. shared parameter gradients can be automatically inferred by theano.
Updating these shared parameters (e.g. within an optimizer) automatically updates also the distribution 'logP', as it
must depend on the very same parameters.


Basic Probabilistic Model
-------------------------
"""

'''
The base probabilistic model would be::

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

            further_references.pop('outputs', None)  # must not be given twice; None is essential as otherwise a Keyerror is thrown if outputs is not given

            super(ProbabilisticModel, self).__init__(
                inputs=inputs,
                outputs=RV,
                RV=RV,  # direct alias
                logP=logP,
                parameters=parameters,
                **further_references
            )
            # could in principal be called before the constructor, however this order seems to make sense for a postmap:
            self.set_postmap(probabilistic_optimizer_postmap)

However, we leav this interface as a convention because when working with substitution, a deterministic model can easily
become a probabilistic one. This suggests that we are working with Models rather than Deterministic/Probabilistic models.

However for the optimizer it is of course crucial, and here the respective optimizer postmap should be used.
'''
# TODO maybe add ProbabilisticModel type which checks for given keys (?)





"""
Noise Models
------------
"""


class DiagGaussianNoise(Model):
    """Class representing a Gaussian with diagnonal covariance matrix.

    Attributes
    ----------

    mean : shared Theano variable wrapping numpy array.
        The mean of the distribution.

    _Var : shared Theano variable wrapping numpy array
        parametrization of var (so that var is always positiv)

    var : Theano variable.
        The variance of the distribution (depends on rho).

    rng : Theano RandomStreams object.
        Random number generator to draw samples from the distribution from.
    """

    def __init__(self, input=None, init_var=None, rng=None):
        """Add a DiagGauss random noise around given input with given variance (defaults to 1).

        Mean default to 0, and var to 1, if not further specified, i.e. standard gaussian random variable.

        Parameters
        ----------

        size : int
            number of outputs
        init_mean : np.array
            means, same length as variances
        init_var : np.array
            variances, same length as means
        rng : Theano RandomStreams object, optional.
            Random number generator to draw samples from the distribution from.
        """
        if input is None:
            input = T.dvector(name="input")

        if init_var is None:
            init_var = T.ones(input.shape, dtype=config.floatX)
        else:
            init_var = np.asarray(init_var, dtype=config.floatX)
            # TODO ensure that input does not get another shape!!!

        self.rng = RandomStreams() if rng is None else rng
        self.var = as_tensor_variable(init_var, "var")  # may use symbolic shared variable

        noise = self.rng.normal(input.shape, dtype=config.floatX)  # everything elementwise # TODO dtype needed?
        outputs = input + T.sqrt(self.var) * noise  # random sampler

        @models_as_outputs
        def logP(rv):
            return (
                (-input.size/2)*T.log(2*np.pi) - (1/2)*T.log(self.var).sum()   # normalizing constant
                - (1 / 2 * T.sum((rv - input) ** 2 / self.var))  # core exponential
            )  # everything is elementwise

        super(DiagGaussianNoise, self).__init__(
            outputs=outputs,
            logP=logP,
            inputs=[input],
            parameters=[],
            parameters_positive=[self.var]
        )


"""
Base Distributions
------------------
"""


class DiagGauss(Model):
    """Class representing a Gaussian with diagnonal covariance matrix.

    Attributes
    ----------

    mean : shared Theano variable wrapping numpy array.
        The mean of the distribution.

    _Var : shared Theano variable wrapping numpy array
        parametrization of var (so that var is always positiv)

    var : Theano variable.
        The variance of the distribution (depends on rho).

    rng : Theano RandomStreams object.
        Random number generator to draw samples from the distribution from.
    """

    def __init__(self, output_size=1, init_mean=None, init_var=None, rng=None):
        """Initialise a DiagGauss random variable with given mean and variance.

        Mean default to 0, and var to 1, if not further specified, i.e. standard gaussian random variable.

        Parameters
        ----------

        output_size : int
            number of outputs
        init_mean : np.array
            means, same length as variances
        init_var : np.array
            variances, same length as means
        rng : Theano RandomStreams object, optional.
            Random number generator to draw samples from the distribution from.
        """
        # parameter preprocessing
        # -----------------------
        # ensure length is the same:
        if init_mean is not None and init_var is not None and len(init_mean) != len(init_var):
            raise ValueError("means and variances need to be of same length")

        if init_mean is not None:
            init_mean = np.asarray(init_mean, dtype=config.floatX)
            output_size = len(init_mean)
        if init_var is not None:
            init_var = np.asarray(init_var, dtype=config.floatX)
            output_size = len(init_var)

        if init_mean is None:
            init_mean = np.zeros(output_size, dtype=config.floatX)
        if init_var is None:
            init_var = np.ones(output_size, dtype=config.floatX)

        # main part
        # ---------
        self.mean = as_tensor_variable(init_mean, "mean")  # TODO broadcastable?
        dgn = DiagGaussianNoise(self.mean, init_var, rng)
        self.var = dgn.var

        kwargs = {'inputs': [], 'parameters': [self.mean]}
        update(kwargs, dgn, overwrite=False)
        super(DiagGauss, self).__init__(**kwargs)


class Uniform(Model):

    def __init__(self, output_size=1, init_start=None, init_offset=None, rng=None):
        """ Initialise a uniform random variable with given range (by start and offset).

        Start default to 0, and offset to 1, if not further specified, i.e. standard uniform random variable.

        Parameters
        ----------

        output_size : int
            number of outputs
        init_start : np.array
            arbitrary, same length as offsets
            gives uniform[start, start+offset]
        init_offset : np.array
            positiv, same length as starts
            gives uniform[start, start+offset]
        rng : Theano RandomStreams object, optional.
            Random number generator to draw samples from the distribution from.
        """
        # ensure length is the same:
        if init_start is not None and init_offset is not None and len(init_start) != len(init_offset):
            raise ValueError("starts and offsets need to be of same length")

        # initialize empty defaults:
        if init_start is None:
            init_start = np.zeros(output_size, dtype=config.floatX)
        else:
            init_start = np.asarray(init_start, dtype=config.floatX)
            output_size = len(init_start)

        if init_offset is None:
            init_offset = np.ones(output_size, dtype=config.floatX)
        else:
            init_offset = np.asarray(init_offset, dtype=config.floatX)
            output_size = len(init_offset)

        if (init_offset <= 0).any():
            raise ValueError("offset must be positive")

        self.rng = RandomStreams() if rng is None else rng

        self.start = as_tensor_variable(init_start, "start")  # TODO broadcastable?
        self.offset = as_tensor_variable(init_offset, "offset")  # TODO broadcastable?

        noise = self.rng.uniform(size=(output_size,), dtype=config.floatX)  # everything elementwise # TODO dtype needed?
        outputs = noise * self.offset + self.start  # random sampler

        @models_as_outputs
        def logP(rv):
            # summed over independend components
            return (T.log(T.le(self.start, rv)) + T.log(T.le(rv, self.start + self.offset)) - T.log(self.offset)).sum()

        super(Uniform, self).__init__(
            outputs=outputs,
            inputs=[],
            logP=logP,
            parameters=[self.start],
            parameters_positive=[self.offset]
        )