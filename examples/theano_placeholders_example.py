#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

import theano.tensor as T
from theano.printing import debugprint

from theano_models import Model, Placeholder

x = T.dscalar('x')
y = T.dscalar('y')

plus = Model([x + y])
minus = Model([x - y])

op_model1 = Placeholder("model1", itypes=[T.dscalar, T.dscalar], otypes=[T.dscalar])
op_model2 = Placeholder("model2", itypes=[T.dscalar, T.dscalar], otypes=[T.dscalar])
print ":: no replacements initially :::::::::::::::::"
print op_model1.replacements
print repr(op_model1)
print op_model1  # this reflects always only the operators name, as it is used within `debugprint`

p1 = op_model1(x,y)
p2 = op_model2(x,y)
p1_p1p2 = op_model1(p1, p2)
print ":: abstract placeholder theano graph :::::::::"
debugprint(p1_p1p2)

print ":: replacing placeholders ::::::::::::::::::::"
op_model1.replace(plus, [p1_p1p2])
debugprint(p1_p1p2)
op_model2.replace(minus, [p1_p1p2])
debugprint(p1_p1p2)

print ":: listed replacements :::::::::::::::::::::::"
print repr(op_model1)
print op_model1.replacements
print repr(op_model2)