#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import theano.tensor as T
from theano import config
from theano.tensor.shared_randomstreams import RandomStreams

from deterministic_models import InvertibleModel
from util.theano_helpers import softplus, softplus_inv, shared
from models import ProbabilisticModel

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

"""
Base Distributions
==================
"""


class DiagGauss(ProbabilisticModel):
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
        # ensure length is the same:
        if init_mean is not None and init_var is not None and len(init_mean) != len(init_var):
            raise ValueError("means and variances need to be of same length")

        # initialize empty defaults:
        if init_mean is None:
            init_mean = np.zeros(output_size, dtype=config.floatX)
        else:
            init_mean = np.asarray(init_mean, dtype=config.floatX)
            output_size = len(init_mean)

        if init_var is None:
            init_var = np.ones(output_size, dtype=config.floatX)
        else:
            init_var = np.asarray(init_var, dtype=config.floatX)
            output_size = len(init_var)

        self.rng = RandomStreams() if rng is None else rng

        self.mean = shared(init_mean, "mean")  # TODO broadcastable?
        self.var = shared(init_var, "var")  # TODO broadcastable?

        noise = self.rng.normal(size=(output_size,), dtype=config.floatX)  # everything elementwise # TODO dtype needed?
        RV = self.mean + T.sqrt(self.var) * noise
        logP = lambda RV: (
            (-output_size / 2) * T.log(2 * np.pi) - (1 / 2) * T.log(self.var).sum()  # normalizing constant
            - (1/2 * T.sum((RV - self.mean) ** 2 / self.var))  # core exponential
        )  # everything is elementwise

        super(DiagGauss, self).__init__(
            RV=RV,
            logP=logP,
            parameters=[self.mean],
            parameters_positive=[self.var]
        )


class Uniform(ProbabilisticModel):

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

        self.start = shared(init_start, "start")  # TODO broadcastable?
        self.offset = shared(init_offset, "offset")  # TODO broadcastable?

        noise = self.rng.uniform(size=(output_size,), dtype=config.floatX)  # everything elementwise # TODO dtype needed?
        RV = noise * self.offset + self.start

        logP = lambda RV: (
            (T.log(T.le(self.start, RV)) + T.log(T.le(RV, self.start + self.offset)) - T.log(self.offset)).sum()  # independend components
        )

        super(Uniform, self).__init__(
            RV=RV,
            logP=logP,
            parameters=[self.start],
            parameters_positive=[self.offset]
        )


"""
Noise Models
============
"""


class GaussianNoise(ProbabilisticModel):
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
        self.var = shared(init_var, name="var")  # may use symbolic shared variable

        noise = self.rng.normal(input.shape, dtype=config.floatX)  # everything elementwise # TODO dtype needed?
        RV = input + T.sqrt(self.var) * noise
        logP = lambda RV: (
            (-input.size/2)*T.log(2*np.pi) - (1/2)*T.log(self.var).sum()   # normalizing constant
            - (1/2 * T.sum((RV - input) ** 2 / self.var))  # core exponential
        )  # everything is elementwise

        super(DiagGauss, self).__init__(
            RV=RV,
            logP=logP,
            inputs=[input],
            parameters=[],
            parameters_positive=[self.var]
        )


"""
Distribution Wrapper
====================

Note, like all wrappers, also these are composable with standard compose function. The only difference is that here
the wrappers are classes and not functions.
"""


class NormalizingFlow(ProbabilisticModel):

    def __init__(self, invertible_model, base_prob_model):
        """ Generic Normalizing Flow

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

        def logP(y):
            return base_prob_model['logP'](invertible_model.inv(y)) - T.log(abs(invertible_model['norm_det']))  # equation (5)

        super(NormalizingFlow, self).__init__(
            RV=invertible_model(base_prob_model['RV']),
            logP=logP,
            parameters=base_prob_model['parameters'] + invertible_model['parameters']
        )
