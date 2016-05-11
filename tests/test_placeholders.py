#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

from theano.printing import debugprint
import theano.tensor as T

from theano_graphs.theano_graphs.theano_placeholders import substitute, ReferencedTheanoGraph, Placeholder

# Example Problem
# ---------------

line = "----------- %s -------------------------------------"

x = T.dscalar('x')
y = T.dscalar('y')
p = x + y
m = x - y


# Test substitutions
# ------------------

s = substitute([p + m], replace={p:m})

print line % "p + m"
debugprint(p + m)
print line % "m + m"
debugprint(m + m)
print line % "substitution"
debugprint(s)


# Test ReferencedTheanoGraph
# --------------------------

refgraph = ReferencedTheanoGraph([x + y])
print line % "refgraph"
print refgraph


# Test placeholders
# -----------------

op_model = Placeholder("model", itypes=[T.dscalar, T.dscalar], otypes=[T.dscalar])
place = op_model(x, y)
print line % "placeholder"
debugprint(place)

op_model.replace(refgraph, [place])
debugprint(place)

print line % "complex placeholder"
op_model2 = Placeholder("model2", itypes=[T.dscalar, T.dscalar], otypes=[T.dscalar])
refgraph2 = ReferencedTheanoGraph([x - y])
p1 = op_model(x,y)
p2 = op_model2(x,y)

p1_p1p2 = op_model(p1, p2)
debugprint(p1_p1p2)
op_model.replace(refgraph2, [p1_p1p2])
debugprint(p1_p1p2)
op_model2.replace(refgraph, [p1_p1p2])
debugprint(p1_p1p2)

print line % "multiple returns"
op_model3 = Placeholder("model3", itypes=[T.dscalar, T.dscalar], otypes=[T.dscalar, T.dscalar])
p3 = op_model3(x,y)

