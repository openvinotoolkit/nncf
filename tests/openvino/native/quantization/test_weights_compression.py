# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import openvino.runtime as ov
import pytest

from nncf.openvino.graph.node_utils import get_const_value
from nncf.quantization import compress_weights
from tests.openvino.native.models import IntegerModel
from tests.openvino.native.models import WeightsModel
from tests.openvino.native.quantization.test_fq_params_calculation import REFERENCE_SCALES_DIR
from tests.shared.helpers import compare_stats
from tests.shared.helpers import load_json

TEST_MODELS = {
    IntegerModel: ["gather_2_data", "matmul_1_data", "matmul_2_data"],
    WeightsModel: ["weights_0", "weights_1"],
}


@pytest.mark.parametrize("model_creator_func", TEST_MODELS)
def test_compress_weights(model_creator_func):
    ref_compressed_weights = TEST_MODELS[model_creator_func]
    model = model_creator_func().ov_model
    compressed_model = compress_weights(model)

    n_compressed_weights = 0
    for op in compressed_model.get_ops():
        if op.get_type_name() == "Constant" and op.get_friendly_name() in ref_compressed_weights:
            assert op.get_element_type() == ov.Type(np.uint8)
            n_compressed_weights += 1

    assert n_compressed_weights == len(ref_compressed_weights)


def test_compare_compressed_weights():
    model = IntegerModel().ov_model
    compressed_model = compress_weights(model)

    def get_next_node(node):
        target_inputs = node.output(0).get_target_inputs()
        assert len(target_inputs) == 1
        next_node = next(iter(target_inputs)).get_node()
        return next_node

    nodes = {}
    ref_compressed_weights = TEST_MODELS[IntegerModel]
    for op in compressed_model.get_ops():
        if op.get_type_name() == "Constant" and op.get_friendly_name() in ref_compressed_weights:
            assert op.get_element_type() == ov.Type(np.uint8)
            compressed_weight = get_const_value(op)

            convert_node = get_next_node(op)
            assert convert_node.get_type_name() == "Convert"

            sub_node = get_next_node(convert_node)
            assert sub_node.get_type_name() == "Subtract"
            zero_point_node = sub_node.input_value(1).get_node()
            zero_point = get_const_value(zero_point_node)

            mul_node = get_next_node(sub_node)
            assert mul_node.get_type_name() == "Multiply"
            scale_node = mul_node.input_value(1).get_node()
            scale = get_const_value(scale_node)

            nodes[op.get_friendly_name()] = {
                "compressed_weight": compressed_weight,
                "zero_point": zero_point,
                "scale": scale,
            }

    ref_stats_path = REFERENCE_SCALES_DIR / "IntegerModel_compressed_weights.json"

    # from tests.shared.helpers import dump_to_json
    # dump_to_json(ref_stats_path, nodes)

    ref_nodes = load_json(ref_stats_path)
    params = ["compressed_weight", "zero_point", "scale"]
    compare_stats(ref_nodes, nodes, params)
