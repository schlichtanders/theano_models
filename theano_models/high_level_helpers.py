#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
from collections import Sequence

import theano
from schlichtanders.mymeta import proxify
from theano import gof
from theano.gof.graph import Variable
from theano.compile import Function

from subgraphs import Subgraph, subgraph_inputs, subgraph_outputs, total_size
from theano_models import Model

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


def fct_to_inputs_outputs(th_graph):
    """ generic helper function to get inputs and outputs from a given ... """
    if isinstance(th_graph, Function):
        outputs = th_graph.maker.fgraph.outputs
        inputs = th_graph.maker.fgraph.inputs
    elif isinstance(th_graph, gof.FunctionGraph):
        outputs = th_graph.outputs
        inputs = th_graph.inputs
    elif isinstance(th_graph, tuple) and len(th_graph) == 2:
        inputs, outputs = th_graph
    elif isinstance(th_graph, Subgraph):
        # inputs, outputs = th_graph['inputs'], th_graph['outputs']
        inputs, outputs = subgraph_inputs(th_graph), subgraph_outputs(th_graph)
    else:
        if isinstance(th_graph, gof.Variable):
            th_graph = [th_graph]
        elif isinstance(th_graph, gof.Apply):
            th_graph = th_graph.outputs
        outputs = th_graph
        inputs = gof.graph.inputs(th_graph)

    if not isinstance(outputs, Sequence):
        outputs = [outputs]

    assert isinstance(inputs, (list, tuple))
    assert isinstance(outputs, (list, tuple))
    assert all(isinstance(v, gof.Variable) for v in inputs + outputs)
    return list(inputs), list(outputs)


def get_equiv_by_name(fct1, fct2):
    """ computes a mapping between to supported graph-formats by assuming unique names """
    inputs1, outputs1 = fct_to_inputs_outputs(fct1)
    inputs2, outputs2 = fct_to_inputs_outputs(fct2)

    fct1_to_name = {}
    for v1 in gof.graph.variables(inputs1, outputs1):
        if v1.name is not None:
            fct1_to_name[v1] = v1.name

    name_to_fct2 = {}
    for v2 in gof.graph.variables(inputs2, outputs2):
        if v2.name is not None:
            name_to_fct2[v2.name] = v2

    return {v1: name_to_fct2[n] for v1, n in fct1_to_name.iteritems()}






