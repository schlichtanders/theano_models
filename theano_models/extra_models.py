#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from schlichtanders.mylists import as_list

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

from base import Model
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from theano import shared, function
from util import clone
from theano.gof.fg import MissingInputError
import theano.tensor as T
from schlichtanders.mymeta import proxify
import warnings


# DEPRECATED, better use PooledRandomStream from util.theano_helpers instead of the standard random number generator
class NoisePool(Model):

    def __init__(self, noise_parameters, pool_size=int(1e8), noise_parameters_source='normal', givens={}):
        self.rng = RandomStreams()
        self.pool = shared(getattr(np.random, noise_parameters_source)(size=pool_size), name="noisepool_"+noise_parameters_source, borrow=True)

        try:  # by default always try to reduce numeric values for shape (as it turns out, theano is currently not able to optimize this by itself)
            shapes = function([], [p.shape for p in noise_parameters], on_unused_input="ignore", givens=givens)()
        except MissingInputError:
            warnings.warn("MissingInputs. Using symbolic version, might be considerably slower. %s" % e)
            # theano does not seem to be able to infer shape without computing random numbers
            # (which this exactly wants to prevent)
            # it turns out that normal random variables rv (they have a .rng attribute e.g.)
            # have rv.owner.inputs[1] = shape(rv), this is extremeley usefuly in our case:
            @as_list
            def shapes():
                for p in noise_parameters:
                    if hasattr(p, 'rng'):
                        yield p.owner.inputs[1]  # TODO implementation detail which may change
                    else:
                        yield clone(p).shape  # seems to calculate random numbers despite not needed

        for p, shape in zip(noise_parameters, shapes):
            size = T.prod(shape)
            start_i = self.rng.random_integers(size=tuple(), low=0, high=pool_size - size - 1) # -1 as also high is inclusive
            new_p = T.reshape(self.pool[start_i:start_i+size], shape)
            new_p.name = (p.name if p.name is not None else str(p)) + "_poolednoise"
            proxify(p, new_p)

        super(NoisePool, self).__init__(
            inputs=[],
            noise=[self.pool],
            outputs=noise_parameters
        )




