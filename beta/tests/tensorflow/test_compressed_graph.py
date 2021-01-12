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

from beta.tests.tensorflow import test_models
from beta.tests.tensorflow.helpers import get_empty_config, create_compressed_model_and_algo_for_test
from beta.tests.tensorflow.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config


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


def check_graph(graph, ref_graph_dir, ref_graph_file_name):
    data_dir = os.path.join(os.path.dirname(__file__), 'data/reference_graphs')
    graph_dir = os.path.join(data_dir, ref_graph_dir)
    graph_path = os.path.abspath(os.path.join(graph_dir, ref_graph_file_name))

    # validate file with graph manually!
    if not os.path.exists(graph_path):
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        tf.io.write_graph(graph, graph_dir, ref_graph_file_name, as_text=False)

    expected_graph = tf.compat.v1.GraphDef()
    with open(graph_path, 'rb') as f:
        expected_graph.ParseFromString(f.read())

    tf.test.assert_equal_graph_def(expected_graph, graph)


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


SPARSITY_ALGORITHMS = [
    'magnitude_sparsity',
]


@pytest.fixture(
    scope='function', params=SPARSITY_ALGORITHMS, ids=SPARSITY_ALGORITHMS
)
def _sparsity_case_config(request):
    sparsity_algorithm = request.param
    graph_dir = os.path.join('sparsity', sparsity_algorithm)
    return SparsityTestCaseConfiguration(graph_dir)


class ModelDesc:
    def __init__(self, pb_filename: str, model_builder, input_sample_sizes):
        self.model_name = self._get_model_name(pb_filename)
        self.model_builder = model_builder
        self.pb_filename = pb_filename
        self.input_sample_sizes = input_sample_sizes

    @staticmethod
    def _get_model_name(dot_filename):
        if isinstance(dot_filename, tuple):
            dot_filename = dot_filename[0]
        return dot_filename[:dot_filename.find('.pb')]


SKIP_MAP = {
    'quantization': {
        'inception_resnet_v2': pytest.mark.skip(reason='gitlab issue #17'),
        'nasnet_mobile': pytest.mark.skip(reason='gitlab issue #18')
    },
    'magnitude_sparsity': {
        'inception_resnet_v2': pytest.mark.skip(reason='gitlab issue #17')
    }
}


def get_test_models_desc(algorithm):
    return [
        ModelDesc('densenet121.pb', test_models.DenseNet121, [1, 32, 32, 3]),
        pytest.param(
            ModelDesc('inception_resnet_v2.pb', test_models.InceptionResNetV2, [1, 75, 75, 3]),
            marks=SKIP_MAP[algorithm].get('inception_resnet_v2', ())
        ),
        ModelDesc('inception_v3.pb', test_models.InceptionV3, [1, 75, 75, 3]),
        ModelDesc('mobilenet_v1.pb', test_models.MobileNet, [1, 128, 128, 3]),
        ModelDesc('mobilenet_v2.pb', test_models.MobileNetV2, [1, 96, 96, 3]),
        pytest.param(
            ModelDesc('nasnet_mobile.pb', test_models.NASNetMobile, [1, 32, 32, 3]),
            marks=SKIP_MAP[algorithm].get('nasnet_mobile', ())
        ),
        ModelDesc('resnet50.pb', test_models.ResNet50, [1, 32, 32, 3]),
        ModelDesc('resnet50_v2.pb', test_models.ResNet50V2, [1, 32, 32, 3]),
        ModelDesc('vgg16.pb', test_models.VGG16, [1, 32, 32, 3]),
        ModelDesc('xception.pb', test_models.Xception, [1, 71, 71, 3]),
        ModelDesc('retinanet.pb', test_models.RetinaNet, [1, None, None, 3]),
        ModelDesc('sequential_model.pb', test_models.SequentialModel, [1, 224, 224, 3]),
        ModelDesc('sequential_no_input_model.pb', test_models.SequentialModelNoInput, [1, 224, 224, 3]),
        ModelDesc('mobilenet_v3_small.pb', test_models.MobileNetV3Small, [1, 32, 32, 3]),
        ModelDesc('shared_layers_model.pb', test_models.SharedLayersModel, [1, 30, 30, 3])
    ]


def keras_model_to_graph_def(model):
    input_signature = []
    for item in model.inputs:
        input_signature.append(tf.TensorSpec(item.shape, item.dtype))
    concrete_function = tf.function(model).get_concrete_function(input_signature)
    return concrete_function.graph.as_graph_def(add_shapes=True)


def remove_control_edges(graph_def):
    for node in graph_def.node:
        inp_names = []
        for inp in node.input:
            if '^' not in inp:
                inp_names.append(inp)

        del node.input[:]
        node.input.extend(inp_names)


def check_model_graph(compressed_model, ref_graph_file_name, ref_graph_dir):
    compressed_graph = keras_model_to_graph_def(compressed_model)
    # remove control edges for a human-readable graph visualization
    # remove_control_edges(compressed_graph)
    check_graph(compressed_graph, ref_graph_dir, ref_graph_file_name)


class TestModelsGraph:
    @pytest.mark.parametrize(
        'desc', get_test_models_desc('quantization'), ids=[
            m.model_name if isinstance(m, ModelDesc)
            else m.values[0].model_name for m in get_test_models_desc('quantization')
        ]
    )
    def test_quantize_network(self, desc: ModelDesc, _quantization_case_config):
        model = desc.model_builder(input_shape=tuple(desc.input_sample_sizes[1:]))
        config = get_basic_quantization_config(_quantization_case_config.qconfig,
                                               input_sample_sizes=desc.input_sample_sizes)
        compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)

        check_model_graph(compressed_model, desc.pb_filename, _quantization_case_config.graph_dir)

    @pytest.mark.parametrize(
        'desc', get_test_models_desc('magnitude_sparsity'), ids=[
            m.model_name if isinstance(m, ModelDesc)
            else m.values[0].model_name for m in get_test_models_desc('magnitude_sparsity')
        ]
    )
    def test_sparsity_network(self, desc: ModelDesc, _sparsity_case_config):
        model = desc.model_builder(input_shape=tuple(desc.input_sample_sizes[1:]))
        config = get_basic_magnitude_sparsity_config(desc.input_sample_sizes)
        config['compression']['params'] = {'schedule': 'multistep'}
        compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)

        check_model_graph(compressed_model, desc.pb_filename, _sparsity_case_config.graph_dir)


QUANTIZE_OUTPUTS = [
    ModelDesc('mobilenet_v2_quantize_outputs.pb', test_models.MobileNetV2, [1, 96, 96, 3]),
    ModelDesc('retinanet_quantize_outputs.pb', test_models.RetinaNet, [1, None, None, 3]),
    ModelDesc('sequential_model_quantize_outputs.pb', test_models.SequentialModel, [1, 224, 224, 3]),
    ModelDesc('shared_layers_model_quantize_outputs.pb', test_models.SharedLayersModel, [1, 30, 30, 3]),
]


@pytest.mark.parametrize('desc', QUANTIZE_OUTPUTS, ids=[m.model_name for m in QUANTIZE_OUTPUTS])
def test_quantize_outputs(desc: ModelDesc, _quantization_case_config):
    model = desc.model_builder(input_shape=tuple(desc.input_sample_sizes[1:]))
    config = get_basic_quantization_config(_quantization_case_config.qconfig,
                                           input_sample_sizes=desc.input_sample_sizes)
    config['compression']['quantize_outputs'] = True
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)

    check_model_graph(compressed_model, desc.pb_filename, _quantization_case_config.graph_dir)
