#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

from numbers import Number

import numpy as np

import theano
import theano.tensor as T
from schlichtanders.myfunctools import fmap
from theano import gradient, gof
from theano import config, grad
from theano.tensor import nlinalg
from itertools import izip

from breze.arch.component import transfer as _transfer

from collections import Sequence

from base import models_as_outputs, model_to_output, inputting_references, outputting_references, Model
from base_tools import softplus
from util import as_tensor_variable, clone, U, merge_key
from placeholders import Placeholder



__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

outputting_references.update(['norm_det'])
inputting_references.update(['parameters', 'parameters_positive'])


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
            input = T.vector()
        if not hasattr(input, 'type') or input.type.broadcastable != (False,):
            raise ValueError("Need singleton input vector.")

        try:
            self.transfer = getattr(_transfer, transfer)
        except (TypeError, AttributeError):
            self.transfer = transfer

        self.weights = T.zeros((input.size, output_size))
        self.weights.name = U("weights")
        self.bias = as_tensor_variable(np.zeros(output_size), U("bias"))

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
            input = T.vector()
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

    def __init__(self, inputs, outputs, parameters, inverse=None, norm_det=None, **further_references):
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
        inverse : function or Model
            inverse of this model
        norm_det : theano expression
            used for variable substitution, e.g. within probability distributions
            can be automatically derived, however sometimes a more efficient version is known
            (alternatively you can improve theanos optimizers)
        """
        # we cannot use ``subgraphs_as_outputs`` because inverse needs to stay Model
        inputs = model_to_output(inputs)
        outputs = model_to_output(outputs)
        if norm_det is None:
            # assume single input to single output # TODO generalize to multiple outputs?
            jac = gradient.jacobian(outputs, inputs[0])
            norm_det = nlinalg.Det()(jac)

        super(InvertibleModel, self).__init__(
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            norm_det=norm_det,
            **further_references
        )

        if inverse is None:
            # make pseudo inverse (will work like a function):
            inverse = Placeholder("inverse", itypes=[outputs.type], otypes=[i.type for i in inputs]) # assumes single output
            inverse.f_inputs = inputs
            inverse.f_outputs = outputs

        if hasattr(inverse, '__call__') and not isinstance(inverse, (Model, theano.OpFromGraph)):
            # TODO currently only single output is supported
            inverse_inputs = [outputs.type()]
            inverse_outputs = inverse(*inverse_inputs)
            inverse = InvertibleModel(
                inputs=inverse_inputs,
                outputs=inverse_outputs,
                inverse=self,
                norm_det=T.inv(norm_det),
                parameters=parameters,
                name="Inverse_" + self.name
            )
        self.inverse = inverse
        InvertibleModel.INVERTIBLE_MODELS.append(self)
        self.is_identity = False

    def check_identity(self):
        change = False
        # already identity function
        # assumes singleton input. Note, using `is` gives error as `proxy is not wrapped` but `proxy == wrapped`
        if self.is_identity and self['inputs'][0] != self.inverse['outputs']:
            raise RuntimeError("Detected that reduced Inverse function was remapped to again work as normal Inverse."
                               "This kind of reversing is not supported yet unfortunately.")
            # TODO: do reversing
            self.is_identity = False
            change = True
        # f(finv(x)) = x  # reduce this, as f(finv(x)) is always used as this in the graph
        # by uniqueness of this model, there can either by f(finv(x)) or finv(f(x)), bot NOT both, (it would result in infinite recursion)
        elif not self.is_identity and not self.inverse.is_identity and self['inputs'][0] == self.inverse['outputs']:
            # make identity function:
            self['outputs'] = self.inverse['inputs']
            # don't change the self['inputs'] as this would change finv which might still be used somewhere else!!,
            self.is_identity = True
            change = True
        # finv(f(x)) = x  # this should NOT be reduced, as f(x) can be used elsewhere in the graph
        return change

    @staticmethod
    def check_all_identities():
        any_change = True  # do while
        while any_change:
            any_change = False
            for im in InvertibleModel.INVERTIBLE_MODELS:
                any_change |= im.check_identity()


class PlanarTransform(InvertibleModel):
    """ invertable transformation, unfortunately without symbolic inverse """

    @models_as_outputs
    def __init__(self, input=None, h=T.tanh, init_w=None, init__u=None, init_b=0, R_to_Rplus=softplus):
        if isinstance(input, Sequence):
            raise ValueError("Need single input variable.")
        if input is None:
            input = T.vector()
        else:
            input = as_tensor_variable(input)

        self.b = as_tensor_variable(init_b, U("b"))
        if init_w is None:
            init_w = T.ones(input.shape)
        self.w = as_tensor_variable(init_w, "w")
        if init__u is None:
            init__u = T.ones(input.shape)
        self._u = as_tensor_variable(init__u, U("_u"))
        # this seems not reversable that easily:
        self.u = self._u + (R_to_Rplus(T.dot(self.w, self._u)) - 1 - T.dot(self.w, self._u)) * self.w / T.dot(self.w, self.w)
        # HINT: this softplus might in fact refer to a simple positive parameter, however the formula seems more complex
        # so I leave it with that
        self.u.name = U("u")

        _inner = T.dot(self.w, input) + self.b
        h = h(_inner)  # make it an theano expression
        dh = grad(h, _inner)

        f = input + self.u * h
        norm_det = 1 + T.dot(self.u, dh*self.w)  # faster than default symbolic norm_det, gives same result, so this should be correct

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
        if isinstance(input, Sequence):
            raise ValueError("Need single input variable.")
        if input is None:
            input = T.vector()
        else:
            input = as_tensor_variable(input)

        self.alpha = as_tensor_variable(init_alpha, U("alpha"))
        self.beta_plus_alpha = as_tensor_variable(init_alpha + init_beta, U("beta+alpha"))
        self.beta = self.beta_plus_alpha - self.alpha
        self.beta.name = U("beta")

        self.z0 = T.zeros(input.shape) if init_z0 is None else as_tensor_variable(init_z0)
        self.z0.name = U("z0")

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
            inverse=inverse,
        )


class LocScaleTransform(InvertibleModel):
    @models_as_outputs
    def __init__(self, input=None, init_loc=None, init_scale=1, independent_scale=False):
        if input is None:
            input = T.vector()
        else:
            input = as_tensor_variable(input)

        if init_loc is None:
            init_loc = T.zeros(input.shape)
        self.loc = as_tensor_variable(init_loc, U("loc"))

        if not independent_scale and not isinstance(init_scale, Number):
            raise ValueError("if not indepent scale, only a scalar variance is needed")
        if independent_scale and isinstance(init_scale, Number):
            init_scale = T.ones(input.shape) * init_scale
        self.scale = as_tensor_variable(init_scale, U("scale"))

        def inverse(y):
            return (y - self.loc)/self.scale

        super(LocScaleTransform, self).__init__(
            inputs=[input],
            outputs=self.loc + self.scale*input,
            parameters=[self.loc],
            parameters_positive=[self.scale],
            inverse=inverse,
            norm_det=self.scale.prod() if independent_scale else self.scale**input.size
        )
