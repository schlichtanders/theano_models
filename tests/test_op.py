#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


import theano_models.util.theano_helpers  # load monkey patches in order to work with default theano code
import theano
import theano.tensor as T

a = T.dvector(name="a")
b = T.dvector(name="b")
c = T.sum(a + b)
c.name="c"

op = theano.OpFromGraph([a], [c])

d = T.dvector(name="d")
result = op(d)
result.name = "results"

print("global", [a,b,c,d], map(hash, [a,b,c,d]))

f = theano.function([b, d], result, on_unused_input="warn")
print(f([1,2,3], [5,5,5]))