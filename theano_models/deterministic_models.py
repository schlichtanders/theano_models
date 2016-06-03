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

from base import Model, models_as_outputs, merge_key
from util import softplus, as_tensor_variable
from placeholders import Placeholder


__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Deterministic Modelling
=======================

Standard Models are deterministic in the sense that they represent functions inputs -> outputs.

Basic Deterministic Model
-------------------------
"""
'''
In combination with referenced parameters, which are optimizable, we in fact already get a well defined
deterministically optimizable model::

    class DeterministicModel(Model):
        """ models prediction of some y (outputs) given some optional xs (inputs)

        hence, loss compares outputs with targets, while outputs depend on some input
        """

        def __init__(self, outputs, parameters, inputs=None, **further_references):
            """ constructs general deterministic model

            Parameters
            ----------
            parameters : list of theano expressions
                parameters to be optimized
            outputs : list of theano operator or Model
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

However, it appears in the context of dynamically changing a model that a deterministic model becomes probabilistic.
What we working with in this setting is always a Model. Therefore we leave this deterministic / probabilistic distinction
as a convention of the respective kwargs.
'''

"""
MLP
---
"""


class AffineNonlinear(Model):

    @models_as_outputs
    def __init__(self, output_size, input=None, transfer='identity'):
        # input needs to be vector
        if input is None:
            input = T.dvector()
        if not hasattr(input, 'type') or input.type.broadcastable != (False,):
            raise ValueError("Need singleton input vector.")

        try:
            self.transfer = getattr(_transfer, transfer)
        except (TypeError, AttributeError):
            self.transfer = transfer

        self.weights = T.zeros((input.size, output_size))
        self.weights.name = "weights"
        self.bias = as_tensor_variable(np.zeros(output_size), "bias")

        output = self.transfer(T.dot(input, self.weights) + self.bias).flatten()  # for some reason I get a matrix here and not a vector

        super(AffineNonlinear, self).__init__(
            inputs=[input],
            outputs=output,
            parameters=[self.weights, self.bias]
        )


class Mlp(Model):

    @models_as_outputs
    def __init__(self, hidden_sizes, output_size, hidden_transfers, output_transfer, input=None):
        if input is None:
            input = T.dvector()
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
            outputs=layer,  # same as layer['outputs'] at this place, as Model automatically handles isinstance(outputs,Model)
            parameters=merge_key(self.layers)
        )


"""
Invertible Models
-----------------

E.g. useful to transform probability distributions. See ``NormalizingFlow`` within ``probabilistic_models``.
"""


class InvertibleModel(Model):

    INVERTIBLE_MODELS = []

    @models_as_outputs
    def __init__(self, inputs, outputs, parameters, inverse_outputs=None, inverse_inputs=None, norm_det=None, **further_references):
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
        inverse_outputs : function (or theano expression, but then 'input_inverse' needs to be given)
            expression which inverses outputs
        inverse_inputs : single theano variable
            input of inverse (if inverse is function, this is automatically constructed)
        norm_det : theano expression
            used for variable substitution, e.g. within probability distributions
            can be automatically derived, however sometimes a more efficient version is known
            (alternatively you can improve theanos optimizers)
        """
        if inverse_outputs is None:
            # make pseudo inverse (will work like a function):
            inverse_outputs = Placeholder("inverse", itypes=[outputs.type], otypes=[i.type for i in inputs])
            inverse_outputs.f_inputs = inputs
            inverse_outputs.f_outputs = outputs

        if hasattr(inverse_outputs, '__call__'):
            # TODO currently only single output is supported
            inverse_inputs = [outputs.type(name="inverse_inputs")]
            inverse_outputs = inverse_outputs(*inverse_inputs)

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
            norm_det=norm_det,
            **further_references
        )
        InvertibleModel.INVERTIBLE_MODELS.append(self)

    @property
    def inv(self):
        return InvertibleModel(
            inputs=self['inverse_inputs'],
            outputs=self['inverse_outputs'],
            inverse_inputs=self['inputs'],
            inverse_outputs=self['outputs'],
            parameters=self['parameters'],
            norm_det=T.inv(self['norm_det']),  # TODO is it really only the inverse?
        )

    def reduce_identity(self):
        changed = False
        # already identity function, ignore this
        if self['inputs'] == [self['outputs']]:
            pass
        # f(finv(x)) = x
        elif self['inputs'] == [self['inverse_outputs']]:
            # make identity function:
            self['outputs'] = self['inverse_inputs']
            self['inputs'] = self['inverse_inputs']  # implies self['inverse_outputs] = self['inverse_inputs']
            changed = True
        # finv(f(x)) = x
        elif self['inverse_inputs'] == [self['outputs']]:
            # make identity function:clone,
            self['inverse_outputs'] = self['inputs']
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

    @models_as_outputs
    def __init__(self, input=None, h=T.tanh, init_w=None, init__u=None, R_to_Rplus=softplus):
        if input is None:
            input = T.dvector(name="z")
        if not hasattr(input, 'type') or input.type.broadcastable != (False,):
            raise ValueError("Need singleton input vector.")

        self.b = as_tensor_variable(0, "b")
        self.w = T.ones(input.shape) if init_w is None else as_tensor_variable(init_w)
        self.w.name = "w"
        self._u = T.zeros(input.shape) if init__u is None else as_tensor_variable(init__u)
        self._u.name = "_u"
        # this seems not reversable that easily:
        self.u = self._u + (R_to_Rplus(T.dot(self.w, self._u)) - 1 - T.dot(self.w, self._u)) * self.w / T.dot(self.w, self.w)
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
            parameters=[self.b, self.w, self._u],
            norm_det=norm_det
        )


class RadialTransform(InvertibleModel):

    @models_as_outputs
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
        if not hasattr(input, 'type') or input.type.broadcastable != (False,):
            raise ValueError("Need singleton input vector.")

        self.alpha = as_tensor_variable(init_alpha, "alpha")
        self.beta_plus_alpha = as_tensor_variable(init_alpha + init_beta, "beta+alpha")
        self.beta = self.beta_plus_alpha - self.alpha
        self.beta.name = "beta"

        self.z0 = T.zeros(input.shape) if init_z0 is None else as_tensor_variable(init_z0)
        self.z0.name = "z0"

        r = (input - self.z0).norm(2)
        h = 1 / (self.alpha + r)
        dh = grad(h, r)

        f = input + self.beta * h * (input - self.z0)
        norm_det = (1 + self.beta*h)**(input.size - 1) * (1 + self.beta * h + self.beta * dh * r)

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
            inverse_outputs=inverse,
        )