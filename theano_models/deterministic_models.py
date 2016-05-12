#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import theano.tensor as T
from theano import gradient, gof
from theano import config, grad
from theano.tensor import nlinalg
from itertools import izip

from breze.arch.component import transfer as _transfer

from base import merge_parameters, Model
from postmaps import deterministic_optimizer_postmap
from util.theano_helpers import softplus, shared
from placeholders import Placeholder


__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Deterministic Modelling
=======================

Standard Graphs are deterministic in the sense that they represent functions inputs -> outputs.

Basic Deterministic Model
-------------------------

In combination with referenced parameters, which are optimizable, we in fact already get a well defined
deterministically optimizable model
"""


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

    def __init__(self, outputs, parameters, inputs=None, **further_references):
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
        self.set_postmap(deterministic_optimizer_postmap)


#: alias
FunctionApproximator = DeterministicModel

"""
MLP
---
"""


class AffineNonlinear(DeterministicModel):

    def __init__(self, output_size, input=None, transfer='identity'):
        # input needs to be vector
        if input is None:
            input = T.dvector()
        elif isinstance(input, Model):
            input = Model['output']
        if not hasattr(input, 'type') or input.type.broadcastable != (False,):
            raise ValueError("Need singleton input vector.")

        try:
            self.transfer = getattr(_transfer, transfer)
        except (TypeError, AttributeError):
            self.transfer = transfer

        self.weights = shared(T.zeros((input.size, output_size)), "weights")
        self.bias = shared(np.zeros(output_size), "bias")

        output = self.transfer(T.dot(input, self.weights) + self.bias)

        super(AffineNonlinear, self).__init__(
            inputs=[input],
            outputs=output,
            parameters=[self.weights, self.bias]
        )


class Mlp(DeterministicModel):

    def __init__(self, hidden_sizes, output_size, hidden_transfers, output_transfer, input=None):
        if input is None:
            input = T.dvector()
        elif isinstance(input, Model):
            input = Model['output']
        if not hasattr(input, 'type') or input.type.broadcastable != (False,):
            raise ValueError("Need singleton input vector.")

        self.layers = []

        # only used for construction, cannot be changed when running, otherwise these are almost like hyperparameters
        output_sizes = hidden_sizes + [output_size]
        transfers = hidden_transfers + [output_transfer]

        layer = input
        for o, t in zip(output_sizes, transfers):
            layer = AffineNonlinear(input=layer, output_size=o, transfer=t)
            self.layers.append(layer)

        super(Mlp, self).__init__(
            inputs=[input],
            outputs=layer,  # same as layer['outputs'] at this place, as Graph automatically handles isinstance(outputs,Graph)
            parameters=merge_parameters(self.layers)
        )


"""
Invertible Models
-----------------

E.g. useful to transform probability distributions. See ``NormalizingFlow`` within ``probabilistic_models``.
"""


class InvertibleModel(DeterministicModel):

    INVERTIBLE_MODELS = []

    def __init__(self, inputs, outputs, parameters, inverse=None, inverse_inputs=None, norm_det=None):
        # TODO support multiple inputs
        """
        Parameters
        ----------
        inputs : single theano expression wrapped in list
            (named inputs and not input to eventually support also list of theano expressions)
        outputs : single theano expression
            like usual
        parameters : list of theano expressions
            like usual
        inverse : function (or theano expression, but then 'input_inverse' needs to be given)
            expression which inverses outputs
        inverse_inputs : single theano variable
            input of inverse (if inverse is function, this is automatically constructed)
        norm_det : theano expression
            used for variable substitution, e.g. within probability distributions
            can be automatically derived, however sometimes a more efficient version is known
            (alternatively you can improve theanos optimizers)
        """
        if inverse is None:
            # make pseudo inverse (will work like a function):
            inverse = Placeholder("inverse", itypes=[outputs.type], otypes=[i.type for i in inputs])
            inverse.f_inputs = inputs
            inverse.f_outputs = outputs

        if hasattr(inverse, '__call__'):
            # TODO currently only single output is supported
            inverse_inputs = [outputs.type(name="inverse_inputs")]
            inverse_outputs = inverse(*inverse_inputs)
        else:
            inverse_outputs = inverse

        inverse_outputs.name = "inverse_outputs"
        inputs[0].name = "f_inputs"
        outputs.name = "f_outputs"

        if norm_det is None:
            # assume single input to single output # TODO generalize to multiple outputs?
            jac = gradient.jacobian(outputs, inputs[0])
            norm_det = nlinalg.Det()(jac)

        super(InvertibleModel, self).__init__(
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            inverse_outputs=inverse_outputs,
            inverse_inputs=inverse_inputs,
            norm_det=norm_det
        )
        InvertibleModel.INVERTIBLE_MODELS.append(self)

    @property
    def inv(self):
        return InvertibleModel(
            inputs=self['inverse_inputs'],
            outputs=self['inverse_outputs'],
            inverse_inputs=self['inputs'],
            inverse=self['outputs'],
            parameters=self['parameters'],
        )

    def reduce_identity(self):
        changed = False
        # already identity function, ignore this
        if self['inputs'] == [self['outputs']]:
            pass
        # f(finv(x)) = x
        elif self['inputs'] == [self['inverse_outputs']]:
            # make identity function:
            self['outputs'] = self['inverse_inputs'][0]
            self['inputs'] = self['inverse_inputs']  # implies self['inverse_outputs] = self['inverse_inputs']
            changed = True
        # finv(f(x)) = x
        elif self['inverse_inputs'] == [self['outputs']]:
            # make identity function:
            self['inverse_outputs'] = self['inputs'][0]
            self['inverse_inputs'] = self['inputs']  # implies self['outputs] = self['inputs']
            changed = True
        return changed

    @staticmethod
    def reduce_all_identities():
        any_change = True  # do while
        while any_change:
            any_change = False
            for im in InvertibleModel.INVERTIBLE_MODELS:
                any_change |= im.reduce_identity()



class PlanarTransform(InvertibleModel):
    """ invertable transformation, unfortunately without symbolic inverse """

    def __init__(self, input=None, h=T.tanh):
        if input is None:
            input = T.dvector(name="z")
        elif isinstance(input, Model):
            input = Model['output']
        if not hasattr(input, 'type') or input.type.broadcastable != (False,):
            raise ValueError("Need singleton input vector.")

        self.b = shared(0, "b")
        self.w = shared(T.ones(input.shape, dtype=config.floatX), "w")
        self._u = shared(T.zeros(input.shape, dtype=config.floatX), "_u")
        # this seems not reversable that easily:
        self.u = self._u + (softplus(T.dot(self.w, self._u)) - 1 - T.dot(self.w, self._u)) * self.w / T.dot(self.w, self.w)
        # HINT: this softplus might in fact refer to a simple positive parameter, however the formula seems more complex
        # so I leave it with that
        self.u.name = "u"

        _inner = T.dot(self.w, input) + self.b
        h = h(_inner)  # make it an theano expression
        dh = grad(h, _inner)

        f = input + self.u * h
        norm_det = 1 + T.dot(self.u, dh*self.w)  # faster than default symbolic norm_det

        super(PlanarTransform, self).__init__(
            inputs=[input],
            outputs=f,
            parameters=[self.b, self.w, self.u],
            norm_det=norm_det
        )


class RadialTransform(InvertibleModel):

    def __init__(self, input=None, init_alpha=1, init_beta=1, init_z0=None):
        """

        Parameters
        ----------
        input : theano variable
        init_alpha : float > 0
        init_beta : float > 0
        """
        # TODO raise error for non-valid init_beta or init_alpha!
        if input is None:
            input = T.dvector(name="z")
        elif isinstance(input, Model):
            input = Model['output']
        if not hasattr(input, 'type') or input.type.broadcastable != (False,):
            raise ValueError("Need singleton input vector.")

        self.alpha = shared(init_alpha, "alpha")
        self.beta_plus_alpha = shared(init_alpha + init_beta, "beta+alpha")
        self.beta = self.beta_plus_alpha - self.alpha
        self.beta.name = "beta"

        if init_z0 is None:
            self.z0 = shared(T.zeros(input.shape))
        else:
            self.z0 = shared(init_z0)
            size = len(init_z0)
        self.z0.name = "z0"

        r = (input - self.z0).norm(2)
        h = 1 / (self.alpha + r)
        dh = grad(h, r)

        f = input + self.beta * h * (input - self.z0)
        norm_det = (1 + self.beta*h)**(size - 1) * (1 + self.beta * h + self.beta * dh * r)

        def inverse(f):
            _lhs = (f - self.z0).norm(2)  # equation (26)
            # solution using sage:
            r = - self.alpha/2 - self.beta/2 + _lhs/2 + 1/2*T.sqrt(
                self.alpha**2 + 2*self.alpha + self.beta**2 + 2*(self.alpha - self.beta)*_lhs + _lhs**2)
            z_ = (f - self.z0) / (r*(1 + self.beta/(self.alpha + r)))
            return self.z0 + r*z_

        super(RadialTransform, self).__init__(
            inputs=[input],
            outputs=f,
            parameters=[self.z0],
            parameters_positive=[self.alpha, self.beta_plus_alpha],  # Constraint for invertibility
            norm_det=norm_det,
            inverse=inverse,
        )