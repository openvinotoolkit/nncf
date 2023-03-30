"""
 Copyright (c) 2023 Intel Corporation
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

from typing import List

import pytest
import numpy as np
import onnx
import json
from copy import deepcopy

from nncf.quantization.algorithms.definitions import OverflowFix
from nncf.onnx.graph.onnx_graph import ONNXGraph

from tests.onnx.conftest import ONNX_TEST_ROOT
from tests.onnx.models import LinearModel
from tests.onnx.quantization.common import min_max_quantize_model


REFERENCE_SCALES_DIR = ONNX_TEST_ROOT / 'data' / 'reference_scales'


def load_json(stats_path):
    with open(stats_path, 'r', encoding='utf8') as json_file:
        return json.load(json_file)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    # pylint: disable=W0221, E0202

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def dump_to_json(local_path, data):
    with open(local_path, 'w', encoding='utf8') as file:
        json.dump(deepcopy(data), file, indent=4, cls=NumpyEncoder)


def compare_stats(expected, actual):
    assert len(expected) == len(actual)
    for ref_name in expected:
        ref_stats = expected[ref_name]
        stats = actual[ref_name]
        ref_scale, ref_zero_point = ref_stats['scale'], ref_stats['zero_point']
        scale, zero_point = stats['scale'], stats['zero_point']

        assert np.allclose(ref_scale, scale, atol=1e-6)
        assert np.allclose(ref_zero_point, zero_point, atol=1e-6)


def get_q_nodes(model: onnx.ModelProto) -> List[onnx.NodeProto]:
    output = {}
    onnx_graph = ONNXGraph(model)
    for node in onnx_graph.get_all_nodes():
        if node.op_type == 'QuantizeLinear':
            scale = onnx_graph.get_initializers_value(node.input[1])
            z_p = onnx_graph.get_initializers_value(node.input[2])
            output[node.name] = {'scale': scale, 'zero_point': z_p}
    return output


@pytest.mark.parametrize('overflow_fix', [OverflowFix.DISABLE, OverflowFix.ENABLE, OverflowFix.FIRST_LAYER],
                         ids=[OverflowFix.DISABLE.value, OverflowFix.ENABLE.value, OverflowFix.FIRST_LAYER.value])
def test_overflow_fix_scales(overflow_fix):
    model = LinearModel()
    quantized_model = min_max_quantize_model(model.onnx_model, quantization_params={'overflow_fix': overflow_fix})
    nodes = get_q_nodes(quantized_model)

    ref_stats_name = model.path_ref_graph.split(".")[0] + f'_overflow_fix_{overflow_fix.value}.json'
    ref_stats_path = REFERENCE_SCALES_DIR / ref_stats_name

    # Unkomment lines below to generate reference for new models.
    dump_to_json(ref_stats_path, nodes)

    ref_nodes = load_json(ref_stats_path)
    compare_stats(ref_nodes, nodes)