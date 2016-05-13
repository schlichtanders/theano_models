#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import izip
import theano.tensor as T
from theano_helpers import as_tensor_variable, clone, is_clonable  # because this is so central
from schlichtanders.mymeta import proxify
from schlichtanders.myfunctools import fmap
from schlichtanders.mylists import remove_duplicates
from schlichtanders.mydicts import update

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

"""
Merge helpers
-------------
"""


def merge_parameters(models, key="parameters"):
    """ combines all params, retaining only SharedVariables """
    parameters = []
    for g in models:
        parameters += g[key]
    # return [p for p in parameters if isinstance(p, SharedVariable)]
    return parameters  # no filtering for SharedVariable possible as everything is theano variable (maybe constant variable)


def merge_inputs(models, key="inputs"):
    """ combines all inputs, retaining only such with empty owner """
    inputs = []
    for g in models:
        inputs += g[key]
    return [i for i in inputs if i.owner is None]


def merge(model_type, *models, **kwargs):
    """ This method is only for convenience and suggestion.
    Short version::
    >>> merged_ = {'parameters':merge_parameters(models), 'inputs':merge_inputs(models)}
    >>> merged = Model(**update(merged_, models[0], overwrite=False)
    or with alternativ ending
    >>> merged = Model(**update(dict(models[0]), merged_))
    which is shorter, but with slight copy overhead.

    Parameters
    ----------
    model_type : model class
        initialized with kwargs
    models : list of Model
        used for further merging
        first model is regarded as base model which additional keys will be used
    merge_rules: dictionary of functions working on models
        mapping merge_key to merger
    """
    merge_rules = kwargs.pop('merge_rules', {'parameters': merge_parameters, 'inputs': merge_inputs})  # python 3 keyword arg alternative

    merged = {k: m(models) for k, m in merge_rules.iteritems()}
    fmap(remove_duplicates, merged)

    update(merged, models[0], overwrite=False)
    return model_type(**merged)

"""
reparameterization helpers
--------------------------
"""


def softplus(x, module=T):
    return module.log(module.exp(x) + 1)


def softplus_inv(y, module=T):
    return module.log(module.exp(y) - 1)


def reparameterize_map(f, finv):
    """
    use e.g. like
    >>> model.map("parameters_positive", reparameterize_map(softplus, softplusinv), "parameters")

    new_param = finv(param)
        param = f(new_param)

    Parameters
    ----------
    f : function theano_variable -> theano_variable
    finv : function theano_variable -> theano_variable

    Returns
    -------
        mappable function
    """
    def reparameterize(param):
        assert is_clonable(param), ("Can only flatten clonable parameters."
                                    "If you used this within ``Model.map`` you probably have to completely reinitialize"
                                    "the model now, as some parameters got proxified, others not")
        cp_param = clone(param)
        cp_param.name = (cp_param.name or str(cp_param)) + "_copy"
        new_param = finv(cp_param)  # clone is decisive as we otherwise get an infinite reference loop
        new_param.name = cp_param.name + "_" + f.func_name
        proxified_param = f(new_param)
        proxified_param.name = (param.name or str(param)) + "_reparam"
        proxify(param, proxified_param)
        return new_param  # instead of old parameter, now refer directly to the new underlying parameter
    return reparameterize

"""
norms and distances
-------------------
"""


def L1(parameters):
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += abs(p).sum()
    return summed_up / n


def L2(parameters):
    summed_up = 0
    n = 0
    for p in parameters:
        n += p.size
        summed_up += (p**2).sum()
    return summed_up / n


def norm_distance(norm=L2):
    def distance(targets, outputs):
        """ targets and outputs are assumed to be *lists* of theano variables """
        return norm([t - o for t, o in izip(targets, outputs)])
    return distance


"""
reshape helpers
---------------
"""


def total_size(variables):
    return sum(v.size for v in variables)


def complex_reshape(vector, variables):
    """ reshapes vector into elements with shapes like variables

    Parameters
    ----------
    vector : list
        shall be reshaped
    variables : list of theano variables
        .size and .shape will be used to reshape vector appropriately

    Returns
    -------
        reshaped parts of the vector
    """
    # NOTE: this only works with ConstantShapeSharedVariables, as usually when ``variables`` get proxified, also the old shape refers to the new variables
    i = 0
    for v in variables:
        yield vector[i:i+v.size].reshape(v.shape)
        i += v.size

