#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import theano
from theano import gof
import theano.tensor as T
import numpy as np
from itertools import izip
from copy import copy

from schlichtanders.mydicts import PassThroughDict, DefaultDict
from schlichtanders.mylists import deepflatten
from schlichtanders.mynumpy import complex_reshape
from schlichtanders.mymeta import proxify

from util import norm_distance, L2, clone, clone_all, as_tensor_variable
from util.theano_helpers import is_clonable, gen_variables, sort_dependent_last, get_dependencies

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


"""
Postmaps
========
Postmaps are meant to be applied just before the model is used somewhere else, e.g. within an optimizer
(Note, until now all postmaps are used for interfacing the optimizers).
As a model is essentially a dictionary, postmaps are mapping a dictionary to a new dictionary. Hence, they are
composable, which is included used in the base implementation of Model.

As the optimizer only needs access to the keys, but won't iterate over it, the ``schlichtanders.mydicts.IdentityDict``
is a well suited type for postmaps to return.

As kwargs of composable functions can be set if the composed function is called, e.g. right before optimization,
further kwargs of the postmaps are very comfortable to set.

"""


"""
Optimizer Postmaps
------------------
"""


def deterministic_optimizer_postmap(model, distance=norm_distance()):
    """ builds premap for a standard deterministic model

    Parameters
    ----------
    model : Model
        to be transformed
    distance : metric, function working on two lists of theano expressions
        comparing targets (given as extra input for optimizer) with outputs
        Defaults to standard square loss.

    Returns
    -------
    IdentityDict over model with standard optimizer keys
    """
    if isinstance(model['outputs'], gof.graph.Variable):
        targets = [model['outputs'].type()]
        outputs = [model['outputs']]
    else:
        targets = [o.type() for o in model['outputs']]
        outputs = model['outputs']

    return PassThroughDict(model,
        loss_inputs=targets + model['inputs'],
        loss=distance(targets, outputs)
    )


def probabilistic_optimizer_postmap(model):
    """ builds premap for a standard probabilistic model

    Parameters
    ----------
    model : Model
        to be transformed

    Returns
    -------
    IdentityDict over model with standard optimizer keys
    """
    # virtual random variable
    # (we cannot use model['RV'] itself, as the automatic gradients will get confused because model['RV'] is a complex sampler)
    RV = model['outputs'].type()  # like targets for deterministic model
    return PassThroughDict(model,
        loss_inputs=[RV] + model['inputs'],
        loss=-model['logP'](RV)
    )


"""
Annealing Postmaps
------------------

Adding a regularizer is a common procedure to improve generalizability of the model (which is usually what we want).
The following define such postmaps.
"""


def regularizer_postmap(model, regularizer_norm=L2, regularizer_scalar=1):
    """ postmap for a standard deterministic model. Simply add this postmap to the model.

    Parameters
    ----------
    model : Model
        kwargs to be adapted by this postmap
    regularizer_norm : function working on list of parameters, returning scalar loss
        shall regularize parameters
    regularizer_scalar : scalar
        weight of loss_regularizer (loss_data is weighted with 1)

    Returns
    -------
    IdentityDict over model
    """
    return PassThroughDict(model,
        loss_data=model['loss'],
        loss_regularizer=regularizer_norm(model['parameters']),
        loss=model['loss'] + regularizer_scalar*regularizer_norm(model['parameters'])
    )


def variational_postmap(model):
    """use this postmap INSTEAD of the standard probabilistic postmap"""
    RV = model['outputs'].type()  # like targets for deterministic model
    return PassThroughDict(model,
        loss_inputs=[RV] + model['inputs'],
        loss=-model['logP'](RV),
        loss_data=-model['loglikelihood'](RV),
        loss_regularizer=1/model['n_data'] * model['kl_prior']
    )



"""
Numerical Postmaps
------------------
"""

'''
def __numerical_parameters(model):
    """ standard remap for accessing shared theano parameters

    if model['parameters'] refers to singleton, then its numerical value is used directly, otherwise the numerical
    values of all parameters are flattened out and concatinated to give a numerical representation alltogether
    """
    num_parameters = []
    # singleton case:
    if len(model['parameters']) == 1:
        # borrow=True is for 1) the case that the optimizer works inplace, 2) its faster
        return model['parameters'][0].get_value(borrow=True)  # return it directly, without packing it into a list
    # else, flatten parameters out (as we cannot guarantee matching shapes):
    else:
        for p in model['parameters']:
            v = p.get_value(borrow=True)  # p.get_value(borrow=True) # TODO what does borrow?
            num_parameters += deepflatten(v)
    return np.array(num_parameters)  # default to numpy type, as this supports numeric operators like indented


def __numericalize(model, loss_reference_name, d_order=0):
    """ numericalizes ``model[loss_reference_name]`` or the respective derivative of order ``d_order``

    It works analogously to ``numerical_parameters`` in that it handles singleton cases or otherwise reshapes flattened
    numerical parameters.

    Parameters
    ----------
    model : Model
        source from which to wrap
    loss_reference_name: str
        model[reference_name] will be wrapped
    d_order : int
        order of derivative (0 stands for no derivative) which shall be computed
    """
    # handle singleton parameters:
    parameters = model['parameters'][0] if len(model['parameters']) == 1 else model['parameters']
    if d_order == 0:
        outputs = model[loss_reference_name]
    elif d_order == 1:
        outputs = theano.grad(model[loss_reference_name], parameters)
    elif d_order == 2:
        try:
            outputs = theano.gradient.hessian(model[loss_reference_name], parameters)
        except AssertionError:
            # TODO hessian breaks if we use constants as parameters (or also matrices), as only vectors are supported
            # possible workaround: adapt the custom shared definition to make constants of broadcastable (True,)
            # instead of ()
            raise TypeError("Cannot (yet) compute hessian for constant parameters.")
    else:
        raise ValueError("Derivative of order %s is not yet implemented" % d_order)

    f_theano = theano.function(model['loss_inputs'], outputs)
    shapes = [p.get_value(borrow=True).shape for p in model['parameters']]  # borrow=True as it is faster

    def f(xs, *args, **kwargs):
        if len(model['parameters']) == 1:
            model['parameters'][0].set_value(xs, borrow=True)  # xs is not packed within list
            return f_theano(*args, **kwargs)  # where initialized correctly

        # else, reshape flattened parameters
        else:
            xs = list(complex_reshape(xs, shapes))
            for x, p in izip(xs, model['parameters']):
                p.set_value(x, borrow=True)

            if d_order == 0:
                return f_theano(*args, **kwargs)
            elif d_order == 1:
                # CAUTION: must return array type, as this output type should be averagable and else...
                return np.array(deepflatten(f_theano(*args, **kwargs)))
            else:
                raise ValueError("flat reparameterization of order %s is not yet implemented" % d_order)
                # other orders not yet implemented (will get a bit messy on the hessian,
                # but there for sure is a general solution to this problem)
    return f
'''

'''
def _numerical_parameters(model, inputs_to_values):
    """ standard remap for accessing shared theano parameters

    if model['parameters'] refers to singleton, then its numerical value is used directly, otherwise the numerical
    values of all parameters are flattened out and concatinated to give a numerical representation alltogether
    """
    # singleton case:
    if len(model['parameters']) == 1:
        # borrow=True is for 1) the case that the optimizer works inplace, 2) its faster
        return model['parameters'][0].eval(inputs_to_values)  # TODO this builds and hashes a compiled version of the subtree

    # else, flatten parameters out (as we cannot guarantee matching shapes):
    num_parameters = []
    for p in model['parameters']:
        v = p.eval(inputs_to_values)  # TODO this builds and hashes a compiled version of the subtree
        num_parameters += deepflatten(v)
    return np.array(num_parameters)  # default to numpy type, as this supports numeric operators like indented


def _numericalize(model, loss_reference_name, d_order=0, num_param_shapes=None):
    """ numericalizes ``model[loss_reference_name]`` or the respective derivative of order ``d_order``

    It works analogously to ``numerical_parameters`` in that it handles singleton cases or otherwise reshapes flattened
    numerical parameters.

    Parameters
    ----------
    model : Model
        source from which to wrap
    loss_reference_name: str
        model[reference_name] will be wrapped
    d_order : int
        order of derivative (0 stands for no derivative) which shall be computed
    """
    # handle singleton parameters:
    parameters = model['parameters'][0] if len(model['parameters']) == 1 else model['parameters']
    if d_order == 0:
        outputs = model[loss_reference_name]
    elif d_order == 1:
        outputs = theano.grad(model[loss_reference_name], parameters)
    elif d_order == 2:
        try:
            outputs = theano.gradient.hessian(model[loss_reference_name], parameters)
        except AssertionError:
            # TODO hessian breaks if we use constants as parameters (or also matrices), as only vectors are supported
            # possible workaround: adapt the custom shared definition to make constants of broadcastable (True,)
            # instead of ()
            raise TypeError("Cannot (yet) compute hessian for constant parameters.")
    else:
        raise ValueError("Derivative of order %s is not yet implemented" % d_order)

    f_theano = theano.function(model['loss_inputs'] + model['parameters'], outputs)

    def f(xs, *args, **kwargs):
        if len(model['parameters']) == 1:
            args += (xs,)  # parameters are initialized like normal inputs
            return f_theano(*args, **kwargs)  # where initialized correctly

        # else, reshape flattened parameters
        else:
            xs = tuple(complex_reshape(xs, num_param_shapes))
            args += xs
            if d_order == 0:
                return f_theano(*args, **kwargs)
            elif d_order == 1:
                # CAUTION: must return array type, as this output type should be averagable and else...
                return np.array(deepflatten(f_theano(*args, **kwargs)))
            else:
                raise ValueError("flat reparameterization of order %s is not yet implemented" % d_order)
                # other orders not yet implemented (will get a bit messy on the hessian,
                # but there for sure is a general solution to this problem)
    return f


def numericalize_postmap(model, annealing=False, wrapper=None, wrapper_kwargs={},
                         save_compiled_functions=True, inputs_to_values=None, adapt_init_params=lambda ps:ps):
    """ postmap to offer an interface for standard numerical optimizer

    'loss' and etc. must be available in the model

    Parameters
    ----------
    model : Model
    annealing : bool
        indicating whether 'loss_data' and 'loss_regularizer' should be used (annealing=True) or 'loss' (default)
    wrapper : function f -> f where f function like used in scipy.optimize.minimize
        wrappers like in schlichtanders.myoptimizers. E.g. batch, online, chunk...
        or a composition of these
    wrapper_kwargs : dict
        extra kwargs for ``wrapper``
    save_compiled_functions : bool
        If false, functions are compiled on every postmap call anew. If true, they are hashed like in a usual DefaultDict
    inputs_to_values : dict
        for parameters which are not grounded, but depend on the input (only needed for initialization)
        NON-OPTIONAL!! (because this hidden behaviour might easily lead to weird bugs)
    adapt_init_params : function numpy-vector -> numpy-vector
        for further control of initial parameters

    Returns
    -------
    DefaultDict over model
    """
    if inputs_to_values is None:
        raise ValueError("Need inputs to values to prevent subtle bugs. If really no inputs are needed, please supply"
                         "empty dictionary {} as kwarg.")
    if wrapper is None:
        if not annealing:
            def wrapper(f, **wrapper_kwargs):
                return f
        else:
            def wrapper(fs, **wrapper_kwargs):
                return sum(fs)

    num_parameters = [p.eval(inputs_to_values) for p in model['parameters']]
    num_shapes = [n.shape for n in num_parameters]
    num_parameters = num_parameters[0] if len(num_parameters) == 1 else np.array(deepflatten(num_parameters))

    d_order = {"num_loss": 0, "num_jacobian": 1, "num_hessian": 2}
    def lazy_numericalize(key):
        if not annealing:
            return _numericalize(model, "loss", d_order=d_order[key])
        else:
            return (
                _numericalize(model, "loss_data", d_order=d_order[key]),
                _numericalize(model, "loss_regularizer", d_order=d_order[key])
            )

    dd = DefaultDict(  # DefaultDict will save keys after they are called the first time
        lambda key: wrapper(lazy_numericalize(key), **wrapper_kwargs),
        num_parameters=num_parameters
    )
    return dd if save_compiled_functions else dd.noexpand()
'''


def flat_numericalize_postmap(model,
                              annealing=False, wrapper=None, wrapper_kwargs={},
                              save_compiled_functions=True, initial_inputs=None, adapt_init_params=lambda ps: ps):
    """ postmap to offer an interface for standard numerical optimizer

    'loss' and etc. must be available in the model

    Parameters
    ----------
    model : Model
    annealing : bool
        indicating whether 'loss_data' and 'loss_regularizer' should be used (annealing=True) or 'loss' (default)
    wrapper : function f -> f where f function like used in scipy.optimize.minimize
        wrappers like in schlichtanders.myoptimizers. E.g. batch, online, chunk...
        or a composition of these
    wrapper_kwargs : dict
        extra kwargs for ``wrapper``
    save_compiled_functions : bool
        If false, functions are compiled on every postmap call anew. If true, they are hashed like in a usual DefaultDict
    initial_inputs : list of values matching model['inputs']
        for parameters which are not grounded, but depend on the input (only needed for initialization)
        NON-OPTIONAL!! (because this hidden behaviour might easily lead to weird bugs)
    adapt_init_params : function numpy-vector -> numpy-vector
        for further control of initial parameters

    Returns
    -------
    DefaultDict over model
    """
    assert len(model['parameters']) == 1
    if initial_inputs is None:
        raise ValueError("Need ``initial_inputs`` to prevent subtle bugs. If really no inputs are needed, please supply"
                         "empty list [] as kwarg.")
    if wrapper is None:
        if not annealing:
            def wrapper(f, **wrapper_kwargs):
                return f
        else:
            def wrapper(fs, **wrapper_kwargs):
                return sum(fs)

    parameters = model['parameters'][0]
    derivatives = {
        "num_loss": lambda loss: model[loss],
        "num_jacobian": lambda loss: theano.grad(model[loss], parameters),
        "num_hessian": lambda loss: theano.gradient.hessian(model[loss], parameters)
    }

    def function(outputs):
        """ compiles function with signature f(params, *loss_inputs) """
        return theano.function([parameters] + model['loss_inputs'], outputs, on_unused_input="warn")

    def numericalize(key):
        try:
            if not annealing:
                return function(derivatives[key]("loss"))
            else:
                return function(derivatives[key]("loss_data")), function(derivatives[key]("loss_regularizer"))
        except (KeyError, TypeError, ValueError):
            raise KeyError("requested key %s not computable" % key)
        except AssertionError:
            # TODO got the following AssertionError which seems to be a bug deep in theano/proxifying theano
            # "Scan has returned a list of updates. This should not happen! Report this to theano-users (also include the script that generated the error)"
            # for now we ignore this
            raise KeyError("Internal Theano AssertionError. Hopefully, this will get fixed in the future.")

    num_parameters = theano.function(model['inputs'], parameters, on_unused_input='warn')(*initial_inputs)
    dd = DefaultDict(  # DefaultDict will save keys after they are called the first time
        default_getitem=lambda key: wrapper(numericalize(key), **wrapper_kwargs),
        default_setitem=lambda key, value: NotImplementedError("You cannot set items on a numericalize postmap."),
        num_parameters=adapt_init_params(num_parameters)
    )
    return dd if save_compiled_functions else dd.noexpand()


"""
Concrete Numeric Optimizer Postmaps
-----------------------------------
"""


def scipy_postmap(model):
    """ extracts kwargs for scipy.optimize.minize as far as available and mandatory

    Parameters
    ----------
    model: Model

    Returns
    -------
    dict
    """
    kwargs = {
        "fun": model["num_loss"],
        "x0": model["num_parameters"],
    }
    try:
        kwargs["jac"] = model["num_jacobian"]
    except KeyError:
        pass

    try:
        kwargs["hessp"] = model["num_hessianp"]
    except KeyError:
        try:
            kwargs["hess"] = model["num_hessian"]
        except KeyError:
            pass
    return kwargs


def climin_postmap(model):
    """ extracts kwargs for climin.util.optimizer as far as available and mandatory

    Parameters
    ----------
    model: Model

    Returns
    -------
    dict
    """
    kwargs = {
        "f": model["num_loss"],
        "wrt": model["num_parameters"],
    }
    try:
        kwargs["fprime"] = model["num_jacobian"]
    except KeyError:
        pass

    return kwargs


"""
Other Postmaps
--------------
"""


def flatten_keys(model, keys_to_flatten=None, initial_inputs=None):
    if keys_to_flatten is None:
        raise ValueError("Need ``key_to_flatten``.")
    if not hasattr(keys_to_flatten, '__iter__'):
        keys_to_flatten = [keys_to_flatten]

    for key in keys_to_flatten:
        try:
            model[key + '_flat']  # another way for testing whether the key is in model or anything passed-through
        except KeyError:
            if initial_inputs is not None:
                flat_sym = T.concatenate([p.flatten() for p in model[key]])
                shapes_sym = [p.shape for p in model[key]]
                f = theano.function(model['inputs'], [flat_sym] + shapes_sym, on_unused_input="warn")
                _f = f(*initial_inputs)
                flat_num, shapes_num = _f[0], _f[1:]
                flat = as_tensor_variable(flat_num)
                flat.name = ":".join((p.name or str(p)) for p in model[key])

                i = 0
                for p, shape in izip(model[key], shapes_num):
                    # for size and shapes we need to refer to the copies, as the original parameters get proxified
                    # (and size/shape refer to the parameters again)
                    size = np.prod(shape)
                    new_p = flat[i:i + size].reshape(shape)
                    new_p.name = (p.name or str(p)) + "_flat"
                    proxify(p, new_p)
                    i += size

                model[key + '_flat'] = flat

            else:
                assert all(is_clonable(p) for p in model[key]), "can only flatten clonable parameters"
                # it is unfortunately not trivial how to flatten parameters
                # one crucial thing is to handle interdependencies of parameters, meaning that p3 could depend on p1
                # while both are parameters finally. If p3 comes before p1, we get that
                # p3? -> flat[p3_slice]? -> p3_cp.shape? -> p1? -> flat[p1_slice]? -> p3_cp.shape?
                # where the last is indeed a pTHREE_cp.shape because p1 comes after p3 and hence needs also p3's shape
                # to get its position in the flat string
                # Fortunately, we can assume that there is no cyclic dependency between parameters as between any
                # well formed theano variables. It is tree-like orderable.

                copies = clone_all(model[key])
                # variables = sort_dependent_last(model[key])
                # copies = []
                # for var in variables:
                #     # replace all nested dependencies with the respective copies (valid by previous sorting)
                #     copies.append(clone(var, replace=dict(izip(variables, copies))))

                # var_to_cp = dict(zip(variables, copies))
                # dependencies = get_dependencies(variables, copies)
                # for var, deps in dependencies.iteritems():
                #     for d in deps:  # this is a variable which owner has var as input
                #         owner_cp = copy(d.owner)
                #         owner_cp.inputs = map(lambda i: var_to_cp.get(i,i), d.owner.inputs)

                flat = T.concatenate([cp.flatten() for cp in copies])
                flat.name = ":".join(cp.name for cp in copies)

                i = 0
                for p, cp in izip(model[key], copies):
                    # for size and shapes we need to refer to the copies, as the original parameters get proxified
                    # (and size/shape refer to the parameters again)
                    new_p = flat[i:i+cp.size].reshape(cp.shape)
                    new_p.name = (p.name or str(p)) + "_flat"
                    proxify(p, new_p)
                    i += cp.size

                model[key + '_flat'] = flat

    return PassThroughDict(model, {
        key: [model[key + '_flat']] for key in keys_to_flatten  # extra nesting as the original key is supposed to be nested
    })


def flatten_parameters(model, initial_inputs=None):
    return flatten_keys(model, keys_to_flatten=["parameters"], initial_inputs=initial_inputs)
