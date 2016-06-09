#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import absolute_import, print_function, division
from itertools import izip

from collections import Sequence, defaultdict
import operator as op

import numpy as np
import os
from functools import reduce
from six import iteritems, itervalues
import six
import shutil

import theano
from theano import gof
from theano.compile.profilemode import ProfileMode
from theano.compile import Function
import pydot as pd
import subprocess

from ..deterministic_models import InvertibleModel
from .. import base
from ..base import Model, Merge
from ..util.theano_helpers import is_pseudo_constant
from ..util import deepflatten_keep_vars
import json

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'
__path__ = os.path.dirname(os.path.realpath(__file__))


CLUSTER_REFERENCE_GROUPS = False  # does not seem to work with current d3 visualization


class MyPyDotFormatter(object):
    """Create `pydot` graph object from Theano function.

    Parameters
    ----------
    compact : bool
        if True, will remove intermediate variables without name.

    Attributes
    ----------
    node_colors : dict
        Color table of node types.
    apply_colors : dict
        Color table of apply nodes.
    shapes : dict
        Shape table of node types.
    """

    def __init__(self, compact=True):
        """Construct PyDotFormatter object."""
        self.compact = compact
        self.node_colors = {'input': 'limegreen',
                            'constant_input': 'SpringGreen',
                            'pseudo_constant_input': 'limegreen',
                            'shared_input': 'YellowGreen',
                            'output': 'dodgerblue',
                            'unused_output': 'lightgrey',
                            'namedvar': 'orange',
                            'inout': 'red'}
        self.apply_colors = {'GpuFromHost': 'red',
                             'HostFromGpu': 'red',
                             'Scan': 'yellow',
                             'Shape': 'cyan',
                             'IfElse': 'magenta',
                             'Elemwise': '#FFAABB',  # dark pink
                             'Subtensor': '#FFAAFF',  # purple
                             'Alloc': '#FFAA22'}  # orange
        self.shapes = {'input': 'box',
                       'output': 'box',
                       'namedvar': 'box',
                       'inout': 'box',
                       'apply': 'ellipse'}
        self.__node_prefix = 'n'

    def __add_node(self, node):
        """Add new node to node list and return unique id.

        Parameters
        ----------
        node : Theano graph node
            Apply node, tensor variable, or shared variable in compute graph.

        Returns
        -------
        str
            Unique node id.
        """
        assert node not in self.__nodes
        _id = '%s%d' % (self.__node_prefix, len(self.__nodes) + 1)
        self.__nodes[node] = _id
        return _id

    def __node_id(self, node):
        """Return unique node id.

        Parameters
        ----------
        node : Theano graph node
            Apply node, tensor variable, or shared variable in compute graph.

        Returns
        -------
        str
            Unique node id.
        """
        if node in self.__nodes:
            return self.__nodes[node]
        else:
            return self.__add_node(node)

    def __call__(self, th_graph, models, dot_graph=None):
        """Create pydot graph from function.

        Parameters
        ----------
        th_graph : theano.compile.function_module.Function
            A compiled Theano function, variable, apply or a list of variables.
        dot_graph: pydot.Dot
            `pydot` graph to which nodes are added. Creates new one if
            undefined.

        Returns
        -------
        pydot.Dot
            Pydot graph of `fct`
        """
        # args preprocessing
        # ==================

        if dot_graph is None:
            dot_graph = pd.Dot()

        self.__nodes = {}

        profile = None
        if isinstance(th_graph, Function):
            mode = th_graph.maker.mode
            if (not isinstance(mode, ProfileMode) or
                    th_graph not in mode.profile_stats):
                mode = None
            if mode:
                profile = mode.profile_stats[th_graph]
            else:
                profile = getattr(th_graph, "profile", None)
            outputs = th_graph.maker.fgraph.outputs
            inputs = th_graph.maker.fgraph.inputs
        elif isinstance(th_graph, gof.FunctionGraph):
            outputs = th_graph.outputs
            inputs = th_graph.inputs
        elif isinstance(th_graph, tuple) and len(th_graph) == 2:
            inputs, outputs = th_graph
        elif isinstance(th_graph, Model):
            # inputs, outputs = th_graph['inputs'], th_graph['outputs']
            inputs, outputs = model_inputs(th_graph), model_outputs(th_graph)
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
        outputs = list(outputs)
        inputs = list(inputs)
        topo = list(outputs + inputs)


        # core parsing
        # ============

        # we handle sub_models independently, as external inputs/outputs need to be mapped to internal ones
        # this is done most straightforwardly by appending this step to the very end, while meanwhile collecting all
        # internal/external relationships respectively
        sub_models = defaultdict(lambda: defaultdict(list))

        for var in topo:
            external_inputs = []
            if isinstance(var, gof.Variable):
                if var in inputs and var in outputs:
                    var_spec_type = 'inout'
                elif isinstance(var, gof.Constant):
                    var_spec_type = 'constant_input'
                elif is_pseudo_constant(var):
                    var_spec_type = 'pseudo_constant_input'
                elif isinstance(var, theano.tensor.sharedvar.TensorSharedVariable):
                    var_spec_type = 'shared_input'
                elif var in inputs or var.owner is None:
                    var_spec_type = 'input'
                else:
                    # variables with parents
                    if hasattr(var, 'clients') and len(var.clients) == 0:
                        var_spec_type = 'unused_output'
                    elif var in outputs:
                        var_spec_type = 'output'
                    else:
                        var_spec_type = 'namedvar'

                    parent, parent_index = get_nested_model(var, models)
                    if not parent:
                        # for sure this is not None as the None case was handled earlier
                        parent, parent_index = var.owner, var.index
                    if parent not in topo:
                        topo.append(parent)

                    # Parent implies Edges
                    # view_map and destroy_map for edges are not yet implemented (see theano.d3viz.formatting for example implementation)
                    var_id = self.__node_id(var)
                    pd_edge = pd.Edge(self.__node_id(parent), var_id, label="%i %s"%(parent_index, type_to_str(var.type)))
                    dot_graph.add_edge(pd_edge)

                    if isinstance(parent, Model):
                        sub_models[parent]['ext_outputs'].append((var, var_id))

                # for all variables do:
                self.make_variable(var, var_spec_type, dot_graph)

            elif isinstance(var, Model):
                # external_inputs = var['inputs']
                external_inputs = model_inputs(var)
                # think about extracting contants here, as they don't refer outside
                # postpone model creation because we need information about external inputs/outputs
                # self.make_nested_model(var, topo, profile, dot_graph)
            elif isinstance(var, gof.Apply):
                self.make_node(var, profile, dot_graph)
                external_inputs = var.inputs
            else:
                raise RuntimeError("var %s instanceof %s should not happen" % (var, var.__class__))

            # edges for Model or Node (external_inputs = [] for gof.Variable case)
            for ext_i in external_inputs:
                if isinstance(var, Model) and ext_i not in inputs and (ext_i.owner is None or is_pseudo_constant(ext_i)):
                    ext_id = None  # skip this external input as it is only confusing
                elif ext_i.owner is None or ext_i.name or ext_i in inputs or is_pseudo_constant(ext_i):  # make extra variable node
                    if ext_i not in topo:
                        topo.append(ext_i)

                    ext_id = self.__node_id(ext_i)
                    pd_edge = pd.Edge(ext_id, self.__node_id(var), label=type_to_str(ext_i.type))
                    dot_graph.add_edge(pd_edge)
                else:  # default to next parent
                    parent, parent_index = get_nested_model(ext_i, models)
                    if not parent:
                        parent, parent_index = ext_i.owner, ext_i.index
                    if parent and parent not in topo:
                        topo.append(parent)

                    ext_id = self.__node_id(parent)
                    pd_edge = pd.Edge(ext_id, self.__node_id(var),
                                      label="%i %s" % (parent_index, type_to_str(ext_i.type)))
                    dot_graph.add_edge(pd_edge)

                if isinstance(var, Model):
                    # they are already sorted by m['inputs'], hence we don't need the reference to the external input itself
                    sub_models[var]['ext_inputs'].append(ext_id)

        # it is important that models from an upper level are not part of the subgraph, otherwise we get recursion,
        # both in this algorithm as well as in the genereal Model layout.
        for m in sub_models:
            models.remove(m)
        print()
        print("####### MODELS ON THIS LEVEL #######")
        for m in sub_models:
            print(m)
        print("####### MODELS ON LEVELS BELOW #######")
        for m in models:
            print(m)

        for m in sub_models:
            self.make_nested_model(m, models=models, profile=profile, dot_graph=dot_graph, **sub_models[m])

        return dot_graph

    def make_node(self, node, profile, dot_graph):
        __node_id = self.__node_id(node)
        nparams = {
            'name': __node_id,
            'label': apply_label(node),
            'profile': apply_profile(node, profile),
            'node_type': 'apply',
            'shape': self.shapes['apply'],
            'apply_op': str(node)
        }
        # nparams['apply_op'] = nparams['label']

        use_color = None
        for opName, color in iteritems(self.apply_colors):
            if opName in node.op.__class__.__name__:
                use_color = color
        if use_color:
            nparams['style'] = 'filled'
            nparams['fillcolor'] = use_color
            nparams['type'] = 'colored'

        pd_node = dict_to_pdnode(nparams)
        dot_graph.add_node(pd_node)

    def make_variable(self, var, var_spec_type, dot_graph):
        var_id = self.__node_id(var)
        var_rough_type = "input" if "input" in var_spec_type else "output" if "output" in var_spec_type else var_spec_type
        vparams = {
            'name': var_id,
            'label': var_label(var),
            'dtype': type_to_str(var.type),
            'tag': var_tag(var),
            'style': 'filled',
            'shape': self.shapes[var_rough_type],
            'node_type': var_spec_type,
            'fillcolor': self.node_colors[var_spec_type],
        }
        pd_var = dict_to_pdnode(vparams)
        dot_graph.add_node(pd_var)

    def make_nested_model(self, m, models, profile, dot_graph, ext_outputs, ext_inputs=[]):  # [] works, as ext_inputs is not modified
        model_id = self.__node_id(m)

        # Model Node on external layer
        mparams = {
            'name': model_id,
            'label': model_label(m),
            'profile': model_profile(m, profile),
            'node_type': 'apply',
            'shape': self.shapes['apply'],
            'apply_op': str(m),
        }
        # mparams['apply_op'] = mparams['label']

        use_color = "yellow"
        for opName, color in iteritems(self.apply_colors):
            if opName in m.__class__.__name__:
                use_color = color
        if use_color:
            mparams['style'] = 'filled'
            mparams['fillcolor'] = use_color
            mparams['type'] = 'colored'

        pd_model = dict_to_pdnode(mparams)
        dot_graph.add_node(pd_model)

        # Subgraph
        subgraph = pd.Cluster(model_id, label=model_label(m))
        gf = MyPyDotFormatter()
        # don't use "." as this leads to relatively complex escaping situation
        # don't use "_" as this is ignored by dot command completely
        gf.__node_prefix = model_id + "n"
        gf(m, models, dot_graph=subgraph)

        dot_graph.add_subgraph(subgraph)
        pd_model.get_attributes()['subg'] = subgraph.get_name()

        if CLUSTER_REFERENCE_GROUPS:  # reference groups
            for k in m:
                if k != "outputs" and isinstance(m[k], Sequence):
                    cluster = pd.Cluster(model_id+"_"+k, label=k, color="blue")
                    for r in m[k]:
                        cluster.add_node(pd.Node(gf.__node_id(r)))
                    subgraph.add_subgraph(cluster)

        # Internal/External mappings
        def format_map(m):
            return str([list(x) for x in m])

        # Inputs mapping
        _inputs = model_inputs(m)
        assert len(ext_inputs) == len(_inputs)
        # not all internal inputs might be shown externally
        ext_int_inputs = [(e, gf.__node_id(i)) for e, i in izip(ext_inputs, _inputs) if e is not None]

        # int_inputs = [gf.__node_id(x) for x in _inputs]
        # assert len(ext_inputs) == len(int_inputs)
        # h = format_map(zip(ext_inputs, int_inputs))
        pd_model.get_attributes()['subg_map_inputs'] = format_map(ext_int_inputs)

        # Output mapping
        hashmap = {hash(v): v_id for v, v_id in ext_outputs}
        _outputs = model_outputs(m)
        ext_outputs = []
        int_outputs = []
        for o in _outputs:
            try:
                ext_outputs.append(hashmap[hash(o)])
                int_outputs.append(gf.__node_id(o))
            except KeyError:  # if the hash is not found, simply only part of the model is referenced outside
                pass
        # ext_outputs = [hashmap[hash(o)] for o in _outputs]
        # int_outputs = [gf.__node_id(x) for x in _outputs]
        assert len(ext_outputs) == len(int_outputs)
        h = format_map(zip(int_outputs, ext_outputs))
        pd_model.get_attributes()['subg_map_outputs'] = h



def var_label(var, precision=3):
    """Return label of variable node."""
    if var.name is not None:
        return var.name
    elif isinstance(var, gof.Constant):
        h = np.asarray(var.data)
        is_const = False
        if h.ndim == 0:
            is_const = True
            h = np.array([h])
        dstr = np.array2string(h, precision=precision)
        if '\n' in dstr:
            dstr = dstr[:dstr.index('\n')]
        if is_const:
            dstr = dstr.replace('[', '').replace(']', '')
        return dstr
    else:
        return type_to_str(var.type)


def var_tag(var):
    """Parse tag attribute of variable node."""
    tag = var.tag
    if hasattr(tag, 'trace') and len(tag.trace) and len(tag.trace[0]) == 4:
        path, line, _, src = tag.trace[0]
        path = os.path.basename(path)
        path = path.replace('<', '')
        path = path.replace('>', '')
        src = src.encode()
        return [path, line, src]
    else:
        return None


def model_inputs(m):
    ret = []
    for r in deepflatten_keep_vars(m[k] for k in m if k in base.inputting_references):
        if r.name is None and (is_pseudo_constant(r)
                               or isinstance(r, gof.Constant)
                               or isinstance(r, theano.tensor.sharedvar.TensorSharedVariable)):
            continue
        ret.append(r)
    return ret


def model_outputs(m):
    return deepflatten_keep_vars(m[k] for k in m if k in base.outputting_references)


def model_label(m):
    """Return label of apply node."""
    # return m.__class__.__name__
    return m.name


def model_profile(m, profile):
    if not profile or profile.fct_call_time == 0:
        return None
    call_time = profile.fct_call_time
    outputs = m['outputs'] if isinstance(m['outputs'], Sequence) else [m['outputs']]
    time = reduce(op.add, (profile.apply_time.get(node, 0) for node in gof.graph.io_toposort(m['inputs'], outputs)))
    return [time, call_time]


def apply_label(node):
    """Return label of apply node."""
    # return str(node)
    return node.op.__class__.__name__


def apply_profile(node, profile):
    """Return apply profiling informaton."""
    if not profile or profile.fct_call_time == 0:
        return None
    call_time = profile.fct_call_time
    time = profile.apply_time.get(node, 0)
    return [time, call_time]


def broadcastable_to_str(b):
    """Return string representation of broadcastable."""
    named_broadcastable = {(): 'scalar',
                           (False,): 'vector',
                           (False, True): 'col',
                           (True, False): 'row',
                           (False, False): 'matrix'}
    if b in named_broadcastable:
        bcast = named_broadcastable[b]
    else:
        bcast = ''
    return bcast


def dtype_to_char(dtype):
    """Return character that represents data type."""
    dtype_char = {
        'complex64': 'c',
        'complex128': 'z',
        'float32': 'f',
        'float64': 'd',
        'int8': 'b',
        'int16': 'w',
        'int32': 'i',
        'int64': 'l'}
    if dtype in dtype_char:
        return dtype_char[dtype]
    else:
        return 'X'


def type_to_str(t):
    """Return str of variable type."""
    if not hasattr(t, 'broadcastable'):
        return str(t)
    s = broadcastable_to_str(t.broadcastable)
    if s == '':
        s = str(t.dtype)
    else:
        s = dtype_to_char(t.dtype) + s
    return s


def dict_to_pdnode(d):
    """Create pydot node from dict."""
    e = dict()
    for k, v in iteritems(d):
        if v is not None:
            if isinstance(v, list):
                v = '\t'.join([str(x) for x in v])
            else:
                v = str(v)
            v = str(v)
            v = v.replace('"', '\'')
            e[k] = v
    pynode = pd.Node(**e)
    return pynode


def get_nested_model(ext_o, models):
    """ Note that this returns the first model found which corresponds to the given external output
    you might need to change the order of models to get the desired result """
    for m in models:
        for k in m:
            # jump over identity models in case of identity is requested:
            if isinstance(m, InvertibleModel) and m.is_identity and k in ['outputs', 'inputs', 'inverse_outputs', 'inverse_inputs']:
                # print("JUMPED OVER IDENTITY MODEL")
                continue
            if k in base.outputting_references:
                outputs = m[k] if isinstance(m[k], Sequence) else [m[k]]
                for i, int_o in enumerate(outputs):
                    if ext_o == int_o:
                        return m, i
    return None, None


"""
d3 vizualization
----------------
"""


def d3viz(th_graph, outfile, models=None, ignore_models=None, copy_deps=True, *args, **kwargs):
    """Create HTML file with dynamic visualizing of a Theano function graph.

    In the HTML file, the whole graph or single nodes can be moved by drag and
    drop. Zooming is possible via the mouse wheel. Detailed information about
    nodes and edges are displayed via mouse-over events. Node labels can be
    edited by selecting Edit from the context menu.

    Input nodes are colored in green, output nodes in blue. Apply nodes are
    ellipses, and colored depending on the type of operation they perform. Red
    ellipses are transfers from/to the GPU (ops with names GpuFromHost,
    HostFromGpu).

    Edges are black by default. If a node returns a view of an
    input, the input edge will be blue. If it returns a destroyed input, the
    edge will be red.

    Parameters
    ----------
    th_graph : theano.compile.function_module.Function
        A compiled Theano function, variable, apply or a list of variables.
    outfile : str
        Path to output HTML file.
    models : list of Model
        become (possibly nested) subgraphs
    copy_deps : bool, optional
        Copy javascript and CSS dependencies to output directory.

    Notes
    -----
    This function accepts extra parameters which will be forwarded to
    :class:`theano.d3viz.formatting.PyDotFormatter`.

    """
    if models is None:
        models = Model.all_models[::-1]  # inverse as latest are probably more complex
    if ignore_models is not None:
        for m in ignore_models:
            models.remove(m)
    for m in Merge.all_merges:
        models.remove(m)

    # Create DOT graph
    formatter = MyPyDotFormatter(*args, **kwargs)
    graph = formatter(th_graph, models)
    # with open('tmp.graph', "w") as f:
    #     f.write(graph.to_string())
    #     subprocess.call(['dot', '-Gnewrank', '-o', 'tmp.dot', 'tmp.graph'])
    # with open('tmp.dot', "r") as f:
    #     dot_graph = f.read()
    dot_graph = graph.create_dot()
    with open("tmp.dot_graph", "w") as f:
        f.write(dot_graph)
    # dot_graph = graph.to_string()
    if not six.PY2:
        dot_graph = dot_graph.decode('utf8')

    # Create output directory if not existing
    outdir = os.path.dirname(outfile)
    if not outdir == '' and not os.path.exists(outdir):
        os.makedirs(outdir)

    # Read template HTML file
    template_file = os.path.join(__path__, 'html', 'template.html')
    with open(template_file) as f:
        template = f.read()

    # Copy dependencies to output directory
    src_deps = __path__
    if copy_deps:
        dst_deps = 'd3viz'
        for d in ['js', 'css']:
            dep = os.path.join(outdir, dst_deps, d)
            if not os.path.exists(dep):
                shutil.copytree(os.path.join(src_deps, d), dep)
    else:
        dst_deps = src_deps

    # Replace patterns in template
    replace = {
        '%% JS_DIR %%': os.path.join(dst_deps, 'js'),
        '%% CSS_DIR %%': os.path.join(dst_deps, 'css'),
        '%% DOT_GRAPH %%': safe_json(dot_graph),
    }
    html = replace_patterns(template, replace)

    # Write HTML file
    with open(outfile, 'w') as f:
        f.write(html)
    # for alternative visualizations
    return dot_graph


def replace_patterns(x, replace):
    """Replace `replace` in string `x`.

    Parameters
    ----------
    s : str
        String on which function is applied
    replace : dict
        `key`, `value` pairs where key is a regular expression and `value` a
        string by which `key` is replaced
    """
    for from_, to in iteritems(replace):
        x = x.replace(str(from_), str(to))
    return x


def safe_json(obj):
    """Encode `obj` to JSON so that it can be embedded safely inside HTML.

    Parameters
    ----------
    obj : object
        object to serialize
    """
    return json.dumps(obj).replace('<', '\\u003c')