# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import List

import openvino.runtime as ov
import pytest

from nncf.common.factory import NNCFGraphFactory
from nncf.common.quantization.quantizer_removal import revert_operations_to_floating_point_precision
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes import openvino_metatypes as ov_metatypes
from nncf.openvino.graph.metatypes.groups import OPERATIONS_WITH_WEIGHTS
from nncf.quantization.advanced_parameters import RestoreMode
from tests.openvino.native.models import LinearQuantizedModel


@dataclass
class InputTestData:
    """
    :param quantized_model: A quantized model in which specified operations need
        to be reverted to floating-point precision.
    :param operations: List of operation names to revert to floating-point precision.
    :param quantizers: List of quantizer names that need to be removed in order to revert
        operations to floating-point precision.
    :param restore_mode: Restore mode.
    :param expected_remaining_quantizers: List of remaining quantizer names.
    """

    quantized_model: ov.Model
    operations: List[str]
    quantizers: List[str]
    restore_mode: RestoreMode
    expected_remaining_quantizers: List[str]


TEST_CASES = [
    InputTestData(
        quantized_model=LinearQuantizedModel().ov_model,
        operations=[
            "MatMul_1",
        ],
        quantizers=[
            "FQ_ReLu_0",
            "FQ_Weights_1",
        ],
        restore_mode=RestoreMode.ACTIVATIONS_AND_WEIGHTS,
        expected_remaining_quantizers=[
            "FQ_Inputs",
            "FQ_Weights_0",
        ],
    ),
    InputTestData(
        quantized_model=LinearQuantizedModel().ov_model,
        operations=["MatMul_0", "MatMul_1"],
        quantizers=[
            "FQ_Inputs",
            "FQ_Weights_0",
            "FQ_ReLu_0",
            "FQ_Weights_1",
        ],
        restore_mode=RestoreMode.ONLY_ACTIVATIONS,
        expected_remaining_quantizers=[
            "FQ_Weights_0",
            "FQ_Weights_1",
        ],
    ),
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_revert_operations_to_floating_point_precision(test_case: InputTestData):
    quantized_model_graph = NNCFGraphFactory.create(test_case.quantized_model)
    operations = [quantized_model_graph.get_node_by_name(name) for name in test_case.operations]
    quantizers = [quantized_model_graph.get_node_by_name(name) for name in test_case.quantizers]

    updated_model = revert_operations_to_floating_point_precision(
        operations,
        quantizers,
        test_case.quantized_model,
        quantized_model_graph,
        test_case.restore_mode,
        [ov_metatypes.OVMatMulMetatype, ov_metatypes.OVEmbeddingMetatype],
        lambda node: node.metatype in OPERATIONS_WITH_WEIGHTS and isinstance(node.layer_attributes, OVLayerAttributes),
        lambda node: node.layer_attributes.get_const_port_ids(),
    )

    updated_model_graph = NNCFGraphFactory.create(updated_model)
    actual_remaining_quantizers = [
        x.node_name for x in updated_model_graph.get_nodes_by_metatypes([ov_metatypes.OVFakeQuantizeMetatype])
    ]

    assert sorted(actual_remaining_quantizers) == sorted(test_case.expected_remaining_quantizers)
