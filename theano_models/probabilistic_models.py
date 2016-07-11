#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

from collections import Sequence
from numbers import Number

import numpy as np
import theano.tensor as T
import theano
from theano import config
from theano.tensor.shared_randomstreams import RandomStreams
from theano.printing import Print

from base import Model, Merge, models_as_outputs, inputting_references, outputting_references
from base_tools import as_merge
from schlichtanders.mydicts import update
from util import as_tensor_variable, U
from itertools import izip

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


#: global random number generator used by default:
RNG = RandomStreams()

inputting_references.update(['parameters', 'parameters_positive', 'parameters_psumto1', 'extra_inputs'])

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

# TODO rebuild all logP to accept multiple valued RV !

class GaussianNoise(Model):
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
    @models_as_outputs
    def __init__(self, input=None, init_var=None, rng=None):
        """Add a DiagGauss random noise around given input with given variance (defaults to 1).

        Mean default to 0, and var to 1, if not further specified, i.e. standard gaussian random variable.

        Parameters
        ----------

        size : int
            number of outputs
        init_mean : np.array
            means, same length as variances
        init_var : scalar
            variance
        rng : Theano RandomStreams object, optional.
            Random number generator to draw samples from the distribution from.
        """
        self.rng = rng or RNG
        if input is None:
            input = T.vector()
        if init_var is None:
            init_var = 1.0
            # TODO ensure that input does not get another shape!!!

        self.var = as_tensor_variable(init_var, "var")  # may use symbolic shared variable

        self.noise = self.rng.normal(input.shape, dtype=config.floatX)  # everything elementwise # TODO dtype needed?
        outputs = input + T.sqrt(self.var) * self.noise  # random sampler

        super(GaussianNoise, self).__init__(
            outputs=outputs,
            inputs=[input],
            parameters=[],
            parameters_positive=[self.var]
        )

        # logP needs to be added after Model creation, as it is kind of an alternative view of the model
        # to support this view, we need to reference self:
        rv = outputs.type("rv")
        # to have a clear interface, that logP always is a function from rv to probability, we remap inputs
        self.logP = Merge(Merge(self, inputs="extra_inputs"),
            name=self.name + ".logP", track=True,
            ignore_references=outputting_references,
            inputs=[rv],
            outputs=(
                - (input.size/2) * (T.log(2*np.pi) + T.log(self.var))   # normalizing constant
                - (1/2 * T.sum((rv - input)**2 / self.var))  # core exponential
            )  # everything is elementwise
         )


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
    @models_as_outputs
    def __init__(self, input=None, init_var=None, rng=None, use_log2=True):
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
        use_log2 : bool
            indicates whether logarithm shall be computed by log2 instead of log
            (on some machines improvement of factor 2 or more for big N)
        """
        self.rng = rng or RNG
        if input is None:
            input = T.vector()
        if init_var is None:
            init_var = T.ones(input.shape, dtype=config.floatX)  # shape refers always to current input, its like T.shape(input)
            # TODO ensure that input does not get another shape!!!

        self.var = as_tensor_variable(init_var, "var")  # may use symbolic shared variable

        self.noise = self.rng.normal(input.shape, dtype=config.floatX)  # everything elementwise # TODO dtype needed?
        outputs = input + T.sqrt(self.var) * self.noise  # random sampler

        super(DiagGaussianNoise, self).__init__(
            outputs=outputs,
            inputs=[input],
            parameters=[],
            parameters_positive=[self.var]
        )

        # logP needs to be added after Model creation, as it is kind of an alternative view of the model
        # to support this view, we need to reference self:
        rv = outputs.type("rv")
        # to have a clear interface, that logP always is a function from rv to probability, extra_inputs
        self.logP = Merge(Merge(self, inputs="extra_inputs"),
            name=self.name + ".logP", track=True,
            ignore_references=outputting_references,
            inputs=[rv],
            outputs=(
                (-input.size / 2) * T.log(2 * np.pi)  # normalizing constant
                - (1/2 * T.log2(self.var).sum()/T.log2(np.e) if use_log2 else 1/2*T.log(self.var).sum())  # ...
                - (1/2 * T.sum((rv - input) ** 2 / self.var))  # core exponential
            )  # everything is elementwise
        )


"""
Base Distributions
------------------
"""


class Gauss(GaussianNoise):
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
    @models_as_outputs
    def __init__(self, output_shape=tuple(), init_mean=None, init_var=None, rng=None):
        """Initialise a DiagGauss random variable with given mean and variance.

        Mean default to 0, and var to 1, if not further specified, i.e. standard gaussian random variable.

        Parameters
        ----------

        output_shape : tuple of int
            shape of output
        init_mean : np.array
            means, same length as variances
        init_var : scalar > 0
            variance
        rng : Theano RandomStreams object, optional.
            Random number generator to draw samples from the distribution from.
        """
        # parameter preprocessing
        # -----------------------
        # ensure length is the same:
        if not isinstance(output_shape, Sequence):
            output_shape = (output_shape,)
        self.output_shape = output_shape
        if init_mean is None:
            init_mean = T.zeros(output_shape, dtype=config.floatX)

        # main part
        # ---------
        self.mean = as_tensor_variable(init_mean, "mean")  # TODO broadcastable?
        super(Gauss, self).__init__(self.mean, init_var, rng=rng)

        # alternative Model Keywords interface:
        self.references['parameters'] += self.references['inputs']
        self.references['inputs'] = []
        self.logP.references['parameters'] += self.logP.references['extra_inputs']
        del self.logP.references['extra_inputs']


class DiagGauss(DiagGaussianNoise):
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
    @models_as_outputs
    def __init__(self, output_size=1, init_mean=None, init_var=None, rng=None, use_log2=True):
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

        if init_var is not None:
            output_size = len(init_var)

        if init_mean is None:
            init_mean = T.zeros((output_size,), dtype=config.floatX)

        # main part
        # ---------
        self.mean = as_tensor_variable(init_mean, "mean")  # TODO broadcastable?
        super(DiagGauss, self).__init__(self.mean, init_var, rng=rng, use_log2=use_log2)

        # alternative Model Keywords interface:
        self.references['parameters'] += self.references['inputs']
        self.references['inputs'] = []
        self.logP.references['parameters'] += self.logP.references['extra_inputs']
        del self.logP.references['extra_inputs']



class Categorical(Model):
        """Class representing a Categorical distribution.

        Attributes
        ----------

        probs: Theano variable
            Has the same shape as the distribution and contains the probability of
            the element being 1 and all others being 0. I.e. rows sum up to 1.

        """
        @models_as_outputs
        def __init__(self, probs, rng=None, eps=1e-8):
            """Initialize a Categorical object.

            Parameters
            ----------

            probs : Theano variable
                Gives the shape of the distribution and contains the probability of
                an element being 1, where only one of a row will be 1. Rows sum up
                to 1.

            rng : Theano RandomStreams object, optional.
                Random number generator to draw samples from the distribution from.
            """
            if isinstance(probs, Number):
                probs = as_tensor_variable(np.ones(probs)/probs)
            self.rng = rng or RNG
            self.probs = probs
            self._probs = T.clip(self.probs, eps, 1 - eps)
            self._probs.name = "clipped probs"

            super(Categorical, self).__init__(
                inputs=[],
                outputs=self.rng.multinomial(pvals=self.probs),
                parameters_psumto1=[probs],
            )
            # logP needs to be added after Model creation, as it is kind of an alternative view of the model
            # to support this view, we need to reference self:
            rv = probs.type("rv")
            self.logP = Merge(self, name=self.name+'.log', ignore_references=outputting_references, track=True,
                inputs=[rv],
                outputs=T.sum(rv * T.log(self._probs))
            )


class Uniform(Model):

    @models_as_outputs
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
        self.rng = rng or RNG
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

        self.start = as_tensor_variable(init_start, "start")  # TODO broadcastable?
        self.offset = as_tensor_variable(init_offset, "offset")  # TODO broadcastable?

        noise = self.rng.uniform(size=(output_size,), dtype=config.floatX)  # everything elementwise # TODO dtype needed?
        outputs = noise * self.offset + self.start  # random sampler

        super(Uniform, self).__init__(
            outputs=outputs,
            inputs=[],
            parameters=[self.start],
            parameters_positive=[self.offset]
        )

        # logP needs to be added after Model creation, as it is kind of an alternative view of the model
        # to support this view, we need to reference self:
        rv = outputs.type("rv")
        self.logP = Merge(self, name=self.name+".logP", ignore_references=outputting_references, track=True,
            inputs=[rv],
            outputs=(T.log(T.le(self.start, rv)) + T.log(T.le(rv, self.start + self.offset)) - T.log(self.offset)).sum()
                    # summed over independend components
        )


"""
Combining Probabilistic Models
------------------------------
"""


class Mixture(Merge):
    def __init__(self, *prob_models, **kwargs):
        """
        Parameters
        ----------
        prob_models: Model
            sub probability models to be combined
        init_mixture_probs: theano vector
            length should be equal to len(prob_models)
            should sum up to 1
        rng: RandomStreams
        kwargs : kwargs
            everything else in kwargs will be passed to the internal Merge
        """
        # TODO assert same length of prob_models and init_mixture_params? theano...
        # initialize to uniform if not given otherwise
        self.mixture_probs = as_tensor_variable(
            kwargs.pop('init_mixture_probs', np.ones((len(prob_models),)) / len(prob_models)),
            "mixture_probs"
        )
        self.rng = kwargs.pop('rng', RNG)

        # sampler
        # -------

        outputs_idx = self.rng.choice(size=tuple(), a=len(prob_models), p=self.mixture_probs)
        outputs_all = T.stack([pm['outputs'] for pm in prob_models])
        outputs = outputs_all[outputs_idx]

        this = dict(
            parameters_psumto1=[self.mixture_probs],
            outputs=outputs
        )
        super(Mixture, self).__init__(this, *prob_models, track=True)

        # logP
        # ----
        rv = prob_models[0].logP['inputs']
        for pm in prob_models[0:]:  # same rv for all (other) models
            pm.logP['inputs'] = rv

        logPs = T.stack([pm.logP['outputs'] for pm in prob_models])  # single vector
        max_logP = T.max(logPs)
        self.logP = Merge(*(pm.logP for pm in prob_models), ignore_references=outputting_references, track=True,
            # input should be rv, as all inputs are rv
            outputs=max_logP + T.log(T.sum(T.exp(logPs - max_logP) * self.mixture_probs))
                    # numerical stable version
        )

        # alternative definition:
        # logPs = [pm.logP['outputs'] for pm in prob_models]
        # ps = [self.mixture_probs[i] for i in xrange(len(logPs))]
        # max_logP = T.largest(*logPs)
        # h = (p * T.exp(lP - max_logP) for p, lP in izip(ps, logPs))
        # logP = max_logP + T.log(T.add(*h))
