"""
 Copyright (c) 2022 Intel Corporation
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
import tensorflow as tf

from nncf.experimental.tensorflow.patch_tf import patch_tf_operations

from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.test_compressed_graph import get_graph_to_layer_var_names_map
from tests.tensorflow.test_compressed_graph import get_model_name
from tests.tensorflow.test_compressed_graph import ModelDesc
from tests.tensorflow.test_compressed_graph import get_basic_quantization_config
from tests.tensorflow.test_compressed_graph import prepare_and_check_graph_def
from tests.tensorflow.test_compressed_graph import prepare_and_check_nx_graph
from tests.tensorflow.test_compressed_graph import QUANTIZERS
from tests.tensorflow.test_compressed_graph import QuantizeTestCaseConfiguration
from tests.tensorflow.test_compressed_graph import create_test_name
from tests.experimental.tensorflow import test_models


patch_tf_operations()


MODELS = [
    ModelDesc('resnet50.pb', test_models.resnet_50, [1, 224, 224, 3]),
]


MODELS_IDS = [
    get_model_name(m) for m in MODELS
]


@pytest.fixture(
    scope='function', params=QUANTIZERS, ids=[create_test_name(quant_params) for quant_params in QUANTIZERS]
)
def _quantization_case_config_v2(request):
    graph_dir = os.path.join('quantized', create_test_name(request.param))
    return QuantizeTestCaseConfiguration(request.param, graph_dir)


def nncf_network_to_tf_graph(nncf_network):
    concrete_function = tf.function(nncf_network).get_concrete_function(nncf_network.input_signature)
    return concrete_function.graph, get_graph_to_layer_var_names_map(concrete_function)


def check_model_graph_v2(compressed_model, ref_graph_filename, ref_graph_dir, rename_resource_nodes):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'reference_graphs')
    graph_dir = os.path.join(data_dir, ref_graph_dir)
    graph_path = os.path.abspath(os.path.join(graph_dir, ref_graph_filename))

    # validate file with graph manually!
    ref_graph_exist = True
    if not os.path.exists(graph_path):
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        ref_graph_exist = False

    compressed_graph, graph_to_layer_var_names_map = nncf_network_to_tf_graph(compressed_model)
    if not rename_resource_nodes:
        graph_to_layer_var_names_map = {}

    ref_graph_ext = os.path.splitext(ref_graph_filename)[1]
    if ref_graph_ext == '.pb':
        prepare_and_check_graph_def(compressed_graph, graph_path, ref_graph_exist,
                                    graph_to_layer_var_names_map)

    else:
        prepare_and_check_nx_graph(compressed_graph, graph_path, ref_graph_exist,
                                   graph_to_layer_var_names_map)


@pytest.mark.parametrize('desc', MODELS, ids=MODELS_IDS)
def test_quantize_network_v2(desc: ModelDesc, _quantization_case_config_v2):
    model = desc.model_builder()

    config = get_basic_quantization_config(_quantization_case_config_v2.qconfig,
                                        input_sample_sizes=desc.input_sample_sizes)
    config['compression']['algorithm'] = 'experimental_quantization'
    if desc.ignored_scopes is not None:
        if 'activations' in config['compression']:
            config['compression']['activations']['ignored_scopes'] = desc.ignored_scopes
        else:
            config['compression']['activations'] = {'ignored_scopes': desc.ignored_scopes}

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)
    check_model_graph_v2(compressed_model, desc.ref_graph_filename, _quantization_case_config_v2.graph_dir,
                        desc.rename_resource_nodes)
