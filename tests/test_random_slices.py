#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import theano.tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'
pool_size = 10
pool = shared(np.random.normal(size=pool_size), name="pool", borrow=True)
param_size = 3

rng = RandomStreams()
i = rng.random_integers(size=tuple(), low=0, high=pool_size - param_size)
i.name="i"

# i = T.as_tensor_variable(4)
pool[i:i+param_size]










