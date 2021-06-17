"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import pytest
from addict import Dict

import tensorflow as tf
import networkx as nx

from tests.tensorflow import test_models
from tests.tensorflow.helpers import get_empty_config, create_compressed_model_and_algo_for_test
from tests.tensorflow.sparsity.magnitude.test_helpers import get_basic_filter_pruning_config
from tests.tensorflow.sparsity.magnitude.test_helpers import get_basic_sparsity_config


def get_basic_quantization_config(qconfig, input_sample_sizes=None):
    config = get_empty_config(input_sample_sizes=input_sample_sizes)
    config['compression'] = {'algorithm': 'quantization',
                             'activations': {
                                 'mode': qconfig.mode,
                                 'per_channel': qconfig.per_channel
                             },
                             'weights': {
                                 'mode': qconfig.mode,
                                 'per_channel': qconfig.per_channel
                             }}
    return config


def get_nx_graph_from_tf_graph(tf_graph: tf.Graph, graph_to_layer_var_names_map: dict):
    def _get_node_attributes(op: tf.Operation):
        attr = {'op': op.type}
        return attr

    def _get_inbound_edges(op: tf.Operation):
        inbound_edges = []
        for input_tensor in op.inputs:
            inbound_edges.append((graph_to_layer_var_names_map.get(input_tensor.op.name, input_tensor.op.name),
                                  graph_to_layer_var_names_map.get(op.name, op.name)))
        return inbound_edges

    nodes = {}
    edges = []
    for op in tf_graph.get_operations():
        op_name = graph_to_layer_var_names_map.get(op.name, op.name)
        nodes[op_name] = _get_node_attributes(op)
        edges.extend(_get_inbound_edges(op))

    nx_graph = nx.DiGraph()

    for node_name, attrs in nodes.items():
        nx_graph.add_node(node_name, **attrs)

    for edge in edges:
        nx_graph.add_edge(*edge)

    return nx_graph


def check_graph_def(graph_def, graph_path: str):
    expected_graph = tf.compat.v1.GraphDef()
    with open(graph_path, 'rb') as f:
        expected_graph.ParseFromString(f.read())

    tf.test.assert_equal_graph_def(expected_graph, graph_def)


def check_nx_graph(nx_graph: nx.DiGraph, graph_path: str):
    expected_graph = nx.drawing.nx_pydot.read_dot(graph_path)

    assert nx_graph.nodes.keys() == expected_graph.nodes.keys()

    for node_name, node_attrs in nx_graph.nodes.items():
        expected_attrs = expected_graph.nodes[node_name]
        assert expected_attrs == node_attrs

    assert nx.DiGraph(expected_graph).edges == nx_graph.edges


class QuantizeTestCaseConfiguration:
    def __init__(self, quant_mode, quant_granularity, graph_dir):
        self.qconfig = Dict()
        self.qconfig.mode = quant_mode
        self.qconfig.per_channel = (quant_granularity == 'per_channel')
        self.graph_dir = graph_dir


QUANTIZERS = [('symmetric', 'per_tensor'), ('asymmetric', 'per_channel')]


@pytest.fixture(
    scope='function', params=QUANTIZERS, ids=['{}_{}'.format(mode, granularity) for mode, granularity in QUANTIZERS]
)
def _quantization_case_config(request):
    quant_mode, quant_granularity = request.param
    graph_dir = os.path.join('quantized', quant_mode, quant_granularity)
    return QuantizeTestCaseConfiguration(quant_mode, quant_granularity, graph_dir)


class SparsityTestCaseConfiguration:
    def __init__(self, graph_dir):
        self.graph_dir = graph_dir


class SparsityAlgo:
    magnitude = 'magnitude_sparsity'
    rb = 'rb_sparsity'


@pytest.fixture(scope='function')
def _magnitude_sparsity_case_config():
    sparsity_algorithm = SparsityAlgo.magnitude
    graph_dir = os.path.join('sparsity', sparsity_algorithm)
    return SparsityTestCaseConfiguration(graph_dir)


@pytest.fixture(scope='function')
def _rb_sparsity_case_config():
    sparsity_algorithm = SparsityAlgo.rb
    graph_dir = os.path.join('sparsity', sparsity_algorithm)
    return SparsityTestCaseConfiguration(graph_dir)


class PruningTestCaseConfiguration:
    def __init__(self, graph_dir):
        self.graph_dir = graph_dir


PRUNING_ALGORITHMS = [
    'filter_pruning',
]


@pytest.fixture(
    scope='function', params=PRUNING_ALGORITHMS, ids=PRUNING_ALGORITHMS
)
def _pruning_case_config(request):
    pruning_algorithm = request.param
    graph_dir = os.path.join('pruning', pruning_algorithm)
    return PruningTestCaseConfiguration(graph_dir)


class ModelDesc:
    def __init__(self, ref_graph_filename: str, model_builder, input_sample_sizes,
                 rename_resource_nodes=False):
        self.model_name, _ = os.path.splitext(ref_graph_filename)
        self.model_builder = model_builder
        self.ref_graph_filename = ref_graph_filename
        self.input_sample_sizes = input_sample_sizes
        self.rename_resource_nodes = rename_resource_nodes


SKIP_MAP = {
    'quantization': {
        'inception_resnet_v2': pytest.mark.skip(reason='gitlab issue #17'),
        'nasnet_mobile': pytest.mark.skip(reason='gitlab issue #18'),
        'mobilenet_v2_slim': pytest.mark.skip(reason='ticket #46349'),
        'xception': pytest.mark.skip(reason='gitlab issue #28')
    },
    'magnitude_sparsity': {
        'inception_resnet_v2': pytest.mark.skip(reason='gitlab issue #17'),
        'nasnet_mobile': pytest.mark.skip(reason='gitlab issue #18'),
        'xception': pytest.mark.skip(reason='gitlab issue #28')
    },
    'filter_pruning': {
        'densenet121': pytest.mark.skip(reason='ticket #50604'),
        'inception_resnet_v2': pytest.mark.skip(reason='gitlab issue #17'),
        'nasnet_mobile': pytest.mark.skip(reason='gitlab issue #18'),
        'xception': pytest.mark.skip(reason='gitlab issue #28'),
        'mask_rcnn': pytest.mark.skip(reason='ticket #50605'),
        'resnet50_v2': pytest.mark.skip(resason='Several masks on one weight'),
        'mobilenet_v2_slim': pytest.mark.skip(reason='ticket #46349')
    },
    'rb_sparsity': {
        'mobilenet_v2_slim': pytest.mark.skip(reason='ticket #46349')
    },
}


def get_test_models_desc(algorithm):
    def ref_name(name):
        # Reason: graph_def change cond_true and cond_false function names
        # regardless tf.compat.v1.reset_default_graph and
        #            tf.keras.backend.clear_session
        ext = '.dot' if algorithm == SparsityAlgo.rb else '.pb'
        return name.split('.')[0] + ext

    # PLEASE USE .dot FORMAT FOR ALL NETS WHICH USE tf.cond OPERATION
    return [
        pytest.param(
            ModelDesc(ref_name('densenet121.pb'), test_models.DenseNet121, [1, 32, 32, 3]),
            marks=SKIP_MAP[algorithm].get('densenet121', ())
        ),
        pytest.param(
            ModelDesc(ref_name('inception_resnet_v2.pb'), test_models.InceptionResNetV2, [1, 75, 75, 3]),
            marks=SKIP_MAP[algorithm].get('inception_resnet_v2', ())
        ),
        ModelDesc(ref_name('inception_v3.pb'), test_models.InceptionV3, [1, 75, 75, 3]),
        ModelDesc(ref_name('mobilenet_v1.pb'), test_models.MobileNet, [1, 128, 128, 3]),
        ModelDesc(ref_name('mobilenet_v2.pb'), test_models.MobileNetV2, [1, 96, 96, 3]),
        pytest.param(
            ModelDesc(ref_name('nasnet_mobile.pb'), test_models.NASNetMobile, [1, 32, 32, 3]),
            marks=SKIP_MAP[algorithm].get('nasnet_mobile', ())
        ),
        ModelDesc(ref_name('resnet50.pb'), test_models.ResNet50, [1, 32, 32, 3]),
        pytest.param(
            ModelDesc(ref_name('resnet50_v2.pb'), test_models.ResNet50V2, [1, 32, 32, 3]),
            marks=SKIP_MAP[algorithm].get('resnet50_v2', ())
        ),
        ModelDesc(ref_name('vgg16.pb'), test_models.VGG16, [1, 32, 32, 3]),
        pytest.param(
            ModelDesc(ref_name('xception.pb'), test_models.Xception, [1, 71, 71, 3]),
            marks=SKIP_MAP[algorithm].get('xception', ())
        ),
        pytest.param(
            ModelDesc(ref_name('retinanet.pb'), test_models.RetinaNet, [1, 603, 603, 3]),
            marks=SKIP_MAP[algorithm].get('retinanet', ())
        ),
        ModelDesc(ref_name('sequential_model.pb'), test_models.SequentialModel, [1, 224, 224, 3]),
        ModelDesc(ref_name('sequential_no_input_model.pb'), test_models.SequentialModelNoInput, [1, 224, 224, 3]),
        pytest.param(
            ModelDesc(ref_name('mobilenet_v3_small.pb'), test_models.MobileNetV3Small, [1, 32, 32, 3]),
            marks=SKIP_MAP[algorithm].get('mobilenet_v3_small', ())
        ),
        pytest.param(
            ModelDesc(ref_name('mobilenet_v3_large.pb'), test_models.MobileNetV3Large, [1, 32, 32, 3]),
            marks=SKIP_MAP[algorithm].get('mobilenet_v3_large', ())
        ),
        pytest.param(
            ModelDesc(ref_name('shared_layers_model.pb'), test_models.SharedLayersModel, [1, 30, 30, 3]),
            marks=SKIP_MAP[algorithm].get('shared_layers_model', ())
        ),
        pytest.param(
            ModelDesc('mask_rcnn.dot', test_models.MaskRCNN, [1, 1024, 1024, 3]),
            marks=SKIP_MAP[algorithm].get('mask_rcnn', ())
        ),
        pytest.param(
            ModelDesc(ref_name('yolo_v4.pb'), test_models.YOLOv4, [1, 603, 603, 3]),
            marks=SKIP_MAP[algorithm].get('yolo_v4', ())
        ),
        pytest.param(
            ModelDesc('mobilenet_v2_slim.dot', test_models.HubMobileNetV2, [1, 224, 224, 3], True),
            marks=SKIP_MAP[algorithm].get('mobilenet_v2_slim', ())
        )
    ]


def get_model_name(desc):
    if isinstance(desc, ModelDesc):
        return desc.model_name
    return desc.values[0].model_name


def keras_model_to_tf_graph(model):
    input_signature = []
    for item in model.inputs:
        input_signature.append(tf.TensorSpec(item.shape, item.dtype))
    concrete_function = tf.function(model).get_concrete_function(input_signature)
    return concrete_function.graph, get_graph_to_layer_var_names_map(concrete_function)


def get_graph_to_layer_var_names_map(concrete_fun):
    names_map = {}
    for layer_var in concrete_fun.variables:
        for value_tensor, graph_name in concrete_fun.graph.captures:
            if layer_var.handle is value_tensor:
                names_map[graph_name.name.split(':')[0]] = layer_var.name.split(':')[0]
    return names_map


def rename_graph_def_nodes(graph_def, names_map: dict):
    for node in graph_def.node:
        node.name = names_map.get(node.name, node.name)
        inp_names = []
        for inp in node.input:
            inp_names.append(names_map.get(inp, inp))
        del node.input[:]
        node.input.extend(inp_names)


def remove_control_edges_from_graph_def(graph_def):
    for node in graph_def.node:
        inp_names = []
        for inp in node.input:
            if '^' not in inp:
                inp_names.append(inp)

        del node.input[:]
        node.input.extend(inp_names)


def prepare_and_check_graph_def(tf_graph: tf.Graph, graph_path: str,
                                ref_graph_exist: bool,
                                graph_to_layer_var_names_map=None,
                                remove_control_edges=False):
    graph_def = tf_graph.as_graph_def(add_shapes=True)
    # remove control edges for a human-readable graph visualization
    if remove_control_edges:
        remove_control_edges_from_graph_def(graph_def)

    if graph_to_layer_var_names_map:
        rename_graph_def_nodes(graph_def, graph_to_layer_var_names_map)

    if not ref_graph_exist:
        graph_dir, ref_graph_filename = os.path.split(graph_path)
        tf.io.write_graph(graph_def, graph_dir, ref_graph_filename, as_text=False)

    check_graph_def(graph_def, graph_path)


def prepare_and_check_nx_graph(tf_graph: tf.Graph, graph_path: str, ref_graph_exist: bool,
                               graph_to_layer_var_names_map: dict):
    nx_graph = get_nx_graph_from_tf_graph(tf_graph, graph_to_layer_var_names_map)

    if os.getenv("NNCF_TEST_REGEN_DOT") is not None and not ref_graph_exist:
        nx.drawing.nx_pydot.write_dot(nx_graph, graph_path)

    check_nx_graph(nx_graph, graph_path)


def check_model_graph(compressed_model, ref_graph_filename, ref_graph_dir, rename_resource_nodes):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'reference_graphs')
    graph_dir = os.path.join(data_dir, ref_graph_dir)
    graph_path = os.path.abspath(os.path.join(graph_dir, ref_graph_filename))

    # validate file with graph manually!
    ref_graph_exist = True
    if not os.path.exists(graph_path):
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        ref_graph_exist = False

    compressed_graph, graph_to_layer_var_names_map = keras_model_to_tf_graph(compressed_model)
    if not rename_resource_nodes:
        graph_to_layer_var_names_map = {}

    ref_graph_ext = os.path.splitext(ref_graph_filename)[1]
    if ref_graph_ext == '.pb':
        prepare_and_check_graph_def(compressed_graph, graph_path, ref_graph_exist,
                                    graph_to_layer_var_names_map)

    else:
        prepare_and_check_nx_graph(compressed_graph, graph_path, ref_graph_exist,
                                   graph_to_layer_var_names_map)


class TestModelsGraph:
    @pytest.mark.parametrize('desc', get_test_models_desc('quantization'), ids=[
        get_model_name(m) for m in get_test_models_desc('quantization')])
    def test_quantize_network(self, desc: ModelDesc, _quantization_case_config):
        model = desc.model_builder(input_shape=tuple(desc.input_sample_sizes[1:]))
        config = get_basic_quantization_config(_quantization_case_config.qconfig,
                                               input_sample_sizes=desc.input_sample_sizes)
        compressed_model, _ = create_compressed_model_and_algo_for_test(model, config, should_init=False)

        check_model_graph(compressed_model, desc.ref_graph_filename, _quantization_case_config.graph_dir,
                          desc.rename_resource_nodes)

    @pytest.mark.parametrize('desc', get_test_models_desc(SparsityAlgo.magnitude), ids=[
        get_model_name(m) for m in get_test_models_desc(SparsityAlgo.magnitude)])
    def test_magnitude_sparsity_network(self, desc: ModelDesc, _magnitude_sparsity_case_config):
        model = desc.model_builder(input_shape=tuple(desc.input_sample_sizes[1:]))
        config = get_basic_sparsity_config(desc.input_sample_sizes, SparsityAlgo.magnitude)
        config['compression']['params'] = {'schedule': 'multistep'}
        compressed_model, _ = create_compressed_model_and_algo_for_test(model, config, should_init=False)

        check_model_graph(compressed_model, desc.ref_graph_filename, _magnitude_sparsity_case_config.graph_dir,
                          desc.rename_resource_nodes)

    @pytest.mark.parametrize('desc', get_test_models_desc(SparsityAlgo.rb), ids=[
        get_model_name(m) for m in get_test_models_desc(SparsityAlgo.rb)])
    def test_rb_sparsity_network(self, desc: ModelDesc, _rb_sparsity_case_config):
        model = desc.model_builder(input_shape=tuple(desc.input_sample_sizes[1:]))
        config = get_basic_sparsity_config(desc.input_sample_sizes, SparsityAlgo.rb)
        config['compression']['params'] = {'schedule': 'multistep'}
        compressed_model, _ = create_compressed_model_and_algo_for_test(model, config, should_init=False)

        check_model_graph(compressed_model, desc.ref_graph_filename, _rb_sparsity_case_config.graph_dir,
                          desc.rename_resource_nodes)

    @pytest.mark.parametrize('desc', get_test_models_desc('filter_pruning'), ids=[
        get_model_name(m) for m in get_test_models_desc('filter_pruning')])
    def test_pruning_network(self, desc: ModelDesc, _pruning_case_config):
        model = desc.model_builder(input_shape=tuple(desc.input_sample_sizes[1:]))
        config = get_basic_filter_pruning_config(desc.input_sample_sizes)
        compressed_model, _ = create_compressed_model_and_algo_for_test(model, config, should_init=False)

        check_model_graph(compressed_model, desc.ref_graph_filename, _pruning_case_config.graph_dir,
                          desc.rename_resource_nodes)


QUANTIZE_OUTPUTS = [
    ModelDesc('mobilenet_v2_quantize_outputs.pb', test_models.MobileNetV2, [1, 96, 96, 3]),
    ModelDesc('retinanet_quantize_outputs.pb', test_models.RetinaNet, [1, 603, 603, 3]),
    ModelDesc('sequential_model_quantize_outputs.pb', test_models.SequentialModel, [1, 224, 224, 3]),
    ModelDesc('shared_layers_model_quantize_outputs.pb', test_models.SharedLayersModel, [1, 30, 30, 3]),
]


@pytest.mark.parametrize('desc', QUANTIZE_OUTPUTS, ids=[m.model_name for m in QUANTIZE_OUTPUTS])
def test_quantize_outputs(desc: ModelDesc, _quantization_case_config):
    model = desc.model_builder(input_shape=tuple(desc.input_sample_sizes[1:]))
    config = get_basic_quantization_config(_quantization_case_config.qconfig,
                                           input_sample_sizes=desc.input_sample_sizes)
    config['compression']['quantize_outputs'] = True
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config, should_init=False)

    check_model_graph(compressed_model, desc.ref_graph_filename, _quantization_case_config.graph_dir,
                      desc.rename_resource_nodes)
