"""
 Copyright (c) 2021 Intel Corporation
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

from nncf.tensorflow.graph.converter import convert_keras_model_to_nncf_graph
from nncf.tensorflow.graph.metatypes.common import get_input_metatypes
from nncf.tensorflow.graph.metatypes.common import get_output_metatypes
from tests.tensorflow.helpers import get_basic_conv_test_model
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.quantization.test_algorithm_quantization import get_basic_quantization_config


def test_struct_auxiliary_nodes_nncf_graph():
    model = get_basic_conv_test_model()
    config = get_basic_quantization_config()
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config, should_init=False)

    nncf_graph = convert_keras_model_to_nncf_graph(compressed_model)

    input_nodes = nncf_graph.get_input_nodes()
    output_nodes = nncf_graph.get_output_nodes()

    assert len(input_nodes) == 1
    assert len(output_nodes) == 1

    assert input_nodes[0].metatype in get_input_metatypes()
    assert output_nodes[0].metatype in get_output_metatypes()
