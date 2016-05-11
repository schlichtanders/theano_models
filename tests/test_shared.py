#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

import theano.tensor as T
from theano import shared, function
from theano import gof

from theano.tensor import shared_randomstreams



state = shared(0)
inc = T.iscalar('inc')
acc = inc + state
accumulator = function([inc], state, updates=[(state, state+inc)])