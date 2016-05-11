#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

class Graph(object):
    def __init__(self, **kwargs):
        print "Graph"
        self.kwargs = kwargs

class Opt(Graph):
    def __init__(self, loss, **kwargs):
        print "Opt"
        super(Opt, self).__init__(loss=loss, **kwargs)

class Prob(Opt):
    def __init__(self, p=None, **kwargs):
        print "Prob"
        super(Prob, self).__init__(loss=p, **kwargs)


class Opt2(Opt):
    def __init__(self, loss1, loss2, loss=None, **kwargs):
        print "Opt2"
        if loss is None:
            loss = loss1 + loss2
        super(Opt2, self).__init__(loss=loss, loss1=loss1, loss2=loss2)

class Var(Prob, Opt2):
    def __init__(self, p, loss1, loss2, **kwargs):
        print "Var"
        super(Var, self).__init__(p=p, loss1=loss1, loss2=loss2, **kwargs)

v = Var(p=1, loss1=10, loss2=100)
# seems to work