#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

import numpy
import theano
import theano.tensor as T
from theano import pp
from theano.tensor.var import TensorVariable

from theano.printing import debugprint
x = T.dscalar('x')
y = x ** 2
gy = T.grad(y, x)
print pp(gy)  # print out the gradient prior to optimization
f = theano.function([x], gy)
print f(4)

print numpy.allclose(f(94.2), 188.4)