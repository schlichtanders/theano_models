#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import theano
from theano import gof
from theano import gradient
from copy import copy, deepcopy
from collections import Sequence
import itertools as itt


__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Placeholder Extension
=====================

On top of the above core implementation, the module supports a ``Placeholder`` class which can build abstract operators
to be used exactly like ``+, -, *, /, **, T.dot, T.log, T.exp, ...``.
These placeholder operators correspond directly to functions, i.e. also to theano ``Model``s.

Placeholder operators can be kept abstract to build up more complex abstract theano expressions. Of course, such
abstract expressions are not executable, so remember to first replace all placeholders with concrete ``Model``s.


Example::

    import theano.tensor as T
    from theano.printing import debugprint
    from theano_placeholders import Model, Placeholder

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
"""


class Placeholder(theano.Op, Sequence):
    """ This is a general placeholder-operator creator for theano.

    It works twofold, firstly it can be used as a normal theano-operator,
    and secondly, it has a list-like interface to access all subsequent replacements
    (the replacements are automatically collected - see ``Placeholder.replace`` for how to elicit an replacement)

    This is useful for prestructuring very big models.
    - You can easily visualizing abstract layers (like visualizing a normal theano expression)
    - You can have placeholders which are abstract in the sense, that depending on your context you replace the
      it with a respective replacement.

    list-like interface::
        - access interface is supported
        - delete interface is supported
        - insert interface is intentionally NOT supported

    Remark::
        This replacement facility is not realizable with the standard interfaces
        ``theano.clone(..., replace=...)`` or ``theano.function(..., givens=...)`` as they replace full variables
        and not operators / Applys.
    """

    # Theano Operator interface
    # -------------------------
    __props__ = ()
    placeholder_grad_suffix = "_grad"

    def __init__(self, name, itypes, otypes):
        """ construct new theano operator

        Parameters
        ----------
        name : str
            name of the operator shown e.g. in ``theano.printing.debugprint()``
        itypes : list of types
            input types to this operator
        otypes : list of types
            output types from this operator
        """
        self.itypes = itypes
        self.otypes = otypes
        self.name = name
        self.replacements = []
        # TODO placeholder gradients!
        self.placeholder_grad = None # gradients are initialized if `grad()` is, otherwise this would result in an infinite recursion here
        super(Placeholder, self).__init__()

    def make_node(self, *inputs):
        assert len(inputs) == len(self.itypes)
        assert all(inp.type == it for inp, it in zip(inputs, self.itypes))
        return theano.Apply(self, inputs, [o() for o in self.otypes])

    def perform(self, node, inputs, output_storage):
        raise NotImplementedError("This is only a placeholder OP. It cannot be executed until replaced.")

    # TODO here further. ASK WHETHER GRADIENTS HAVE ALWAYS SAME TYPE SIGNATURE AS MAIN OPERATOR
    # use theano.grad(cost=None, wrt=model[inputs], known_gradients = OrderedDict(zip(model[outputs], output_gradients)))
    '''
    def grad(self, inputs, output_gradients):
        """ returns symbolic placeholder gradient """

        self.placeholder_grad = Placeholder(name=self.name + Placeholder.placeholder_grad_suffix,
                        itypes=[ #TODO combine types of inputs and output_gradients],
                        otypes=self.otypes)
        return [pl_grad(inputs + output_storage) for pl_grad in self.placeholder_grad]
    '''


    # Placeholder logic
    # -----------------

    def replace(self, model, outputs, inputs=None, collecting_replacements=True):
        """ This replaces every occurrence of the placeholder operator in outputs with a copy of refgraph

        Parameters
        ----------
        model : Model
            placeholder replacement (will be copied)
        outputs: list of theano expressions
            specifies theano graph where to replace placeholder (inplace)
        inputs: optional list of theano expressions
            defaults to ``theano.gof.graph.inputs(outputs)``. Specifies theano graph further.
        collecting_replacements: bool
            flag denoting whether replacements shall be collected in self (True)
        """
        inputs = gof.graph.inputs(outputs) if inputs is None else inputs

        new_models = []

        # check same operator inputs/outputs
        assert len(self.itypes) == len(model['inputs'])
        assert all(inp.type == it for inp, it in zip(model['inputs'], self.itypes))

        assert len(self.otypes) == len(model['outputs'])
        assert all(out.type == ot for out, ot in zip(model['outputs'], self.otypes))

        for apply in gof.graph.list_of_nodes(inputs, outputs):
            if apply.op is self:
                model_copy = deepcopy(model)

                model_copy.substitute_key('inputs', apply.inputs)
                # TODO BUG - owner.outputs are not updated
                # apply.outputs are all within one list, however model_copy['outputs'] might refer to several
                # different owners (Applys)

                grouped_by_owner = itt.groupby(
                    zip(apply.outputs, model_copy['outputs']),
                    key=lambda t:t[1].owner
                )
                for owner, owner_outputs in grouped_by_owner:
                    apply_outs, model_outs = zip(*owner_outputs)
                    for o in apply_outs:
                        o.owner = owner
                    owner.outputs = apply_outs
                # change outputs in model to point to old_o
                model_copy['outputs'] = apply.outputs

                new_models.append(model_copy)

        if collecting_replacements:
            self.replacements += new_models
        return new_models


    # Sequence interface
    # ------------------

    def __getitem__(self, index):
        return self.replacements[index]

    def __iter__(self):
        if hasattr(self, 'replacements'):
            return iter(self.replacements)
        else:
            raise StopIteration

    def __len__(self):
        if hasattr(self, 'replacements'):
            return len(self.replacements)
        else:
            return 0

    # list-like delete interface (copied from collections.MutableSequence)
    # --------------------------

    def __delitem__(self, index):
        del self.replacements[index]

    def pop(self, index=-1):
        '''S.pop([index]) -> item -- remove and return item at index (default last).
           Raise IndexError if list is empty or index is out of range.
        '''
        v = self[index]
        del self[index]
        return v

    def remove(self, value):
        '''S.remove(value) -- remove first occurrence of value.
           Raise ValueError if the value is not present.
        '''
        del self[self.index(value)]

    # visualizing interface
    # ---------------------

    def __str__(self):
        # this must be kept simple, as `str(operator)` is called for `theano.printing.debugprint()`
        return self.name

    def __repr__(self):
        return "%s:[%s]" % (self.name, ",".join(str(r) for r in self.replacements))