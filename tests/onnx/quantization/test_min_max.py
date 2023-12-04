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
from dataclasses import dataclass
from typing import List

import pytest

import nncf.onnx.graph.metatypes.onnx_metatypes as om
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.onnx.graph.nncf_graph_builder import ONNXLayerAttributes
from nncf.onnx.graph.node_utils import get_quantization_axis
from nncf.onnx.graph.node_utils import get_reduction_shape
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint


@dataclass
class TestCase:
    nncf_node: NNCFNode
    target_point: ONNXTargetPoint
    per_channel: bool
    ref_reduction_shape: List[int]


test_cases = (
    TestCase(
        nncf_node=NNCFNode(
            {
                NNCFNode.ID_NODE_ATTR: 0,
                NNCFNode.NODE_NAME_ATTR: "conv_with_weight_per_tensor",
                NNCFNode.METATYPE_ATTR: om.ONNXConvolutionMetatype,
                NNCFNode.LAYER_ATTRIBUTES: ONNXLayerAttributes(weight_attrs={1: {"shape": [3, 5, 8]}}),
            }
        ),
        target_point=ONNXTargetPoint(
            target_type=TargetType.OPERATION_WITH_WEIGHTS,
            target_node_name="conv_with_weight_per_tensor",
            port_id=1,
        ),
        per_channel=False,
        ref_reduction_shape=None,
    ),
    TestCase(
        nncf_node=NNCFNode(
            {
                NNCFNode.ID_NODE_ATTR: 0,
                NNCFNode.NODE_NAME_ATTR: "conv_with_weight_per_channel",
                NNCFNode.METATYPE_ATTR: om.ONNXConvolutionMetatype,
                NNCFNode.LAYER_ATTRIBUTES: ONNXLayerAttributes(weight_attrs={1: {"shape": [3, 5, 8]}}),
            }
        ),
        target_point=ONNXTargetPoint(
            target_type=TargetType.OPERATION_WITH_WEIGHTS,
            target_node_name="gemm_with_weight_per_channel_0_port",
            port_id=1,
        ),
        per_channel=True,
        ref_reduction_shape=(1, 2),
    ),
    TestCase(
        nncf_node=NNCFNode(
            {
                NNCFNode.ID_NODE_ATTR: 0,
                NNCFNode.NODE_NAME_ATTR: "gemm_with_weight_per_tensor",
                NNCFNode.METATYPE_ATTR: om.ONNXGemmMetatype,
                NNCFNode.LAYER_ATTRIBUTES: ONNXLayerAttributes(weight_attrs={1: {"shape": [5, 8]}}),
            }
        ),
        target_point=ONNXTargetPoint(
            target_type=TargetType.OPERATION_WITH_WEIGHTS,
            target_node_name="gemm_with_weight_per_tensor",
            port_id=1,
        ),
        per_channel=False,
        ref_reduction_shape=None,
    ),
    TestCase(
        nncf_node=NNCFNode(
            {
                NNCFNode.ID_NODE_ATTR: 0,
                NNCFNode.NODE_NAME_ATTR: "gemm_with_weight_per_channel",
                NNCFNode.METATYPE_ATTR: om.ONNXGemmMetatype,
                NNCFNode.LAYER_ATTRIBUTES: ONNXLayerAttributes(weight_attrs={1: {"shape": [5, 8]}}),
            }
        ),
        target_point=ONNXTargetPoint(
            target_type=TargetType.OPERATION_WITH_WEIGHTS,
            target_node_name="gemm_with_weight_per_channel_0_port",
            port_id=1,
        ),
        per_channel=True,
        ref_reduction_shape=(0,),
    ),
    TestCase(
        nncf_node=NNCFNode(
            {
                NNCFNode.ID_NODE_ATTR: 0,
                NNCFNode.NODE_NAME_ATTR: "gemm_with_weight_per_channel_extra_attrs",
                NNCFNode.METATYPE_ATTR: om.ONNXGemmMetatype,
                NNCFNode.LAYER_ATTRIBUTES: ONNXLayerAttributes(
                    weight_attrs={1: {"shape": [5, 8]}}, node_attrs={"transA": 0, "transB": 0}
                ),
            }
        ),
        target_point=ONNXTargetPoint(
            target_type=TargetType.OPERATION_WITH_WEIGHTS,
            target_node_name="gemm_with_weight_per_channel_extra_attrs",
            port_id=1,
        ),
        per_channel=True,
        ref_reduction_shape=(0,),
    ),
    TestCase(
        nncf_node=NNCFNode(
            {
                NNCFNode.ID_NODE_ATTR: 0,
                NNCFNode.NODE_NAME_ATTR: "gemm_with_weight_per_channel_extra_attrs",
                NNCFNode.METATYPE_ATTR: om.ONNXGemmMetatype,
                NNCFNode.LAYER_ATTRIBUTES: ONNXLayerAttributes(
                    weight_attrs={1: {"shape": [5, 8]}}, node_attrs={"transA": 1, "transB": 0}
                ),
            }
        ),
        target_point=ONNXTargetPoint(
            target_type=TargetType.OPERATION_WITH_WEIGHTS,
            target_node_name="gemm_with_weight_per_channel_extra_attrs",
            port_id=1,
        ),
        per_channel=True,
        ref_reduction_shape=(0,),
    ),
    TestCase(
        nncf_node=NNCFNode(
            {
                NNCFNode.ID_NODE_ATTR: 0,
                NNCFNode.NODE_NAME_ATTR: "gemm_with_weight_per_channel_transpose",
                NNCFNode.METATYPE_ATTR: om.ONNXGemmMetatype,
                NNCFNode.LAYER_ATTRIBUTES: ONNXLayerAttributes(
                    weight_attrs={1: {"shape": [5, 8]}}, node_attrs={"transA": 0, "transB": 1}
                ),
            }
        ),
        target_point=ONNXTargetPoint(
            target_type=TargetType.OPERATION_WITH_WEIGHTS,
            target_node_name="gemm_with_weight_per_channel_transpose",
            port_id=1,
        ),
        per_channel=True,
        ref_reduction_shape=(1,),
    ),
    TestCase(
        nncf_node=NNCFNode(
            {
                NNCFNode.ID_NODE_ATTR: 0,
                NNCFNode.NODE_NAME_ATTR: "gemm_with_weight_per_channel_transpose_one_dim",
                NNCFNode.METATYPE_ATTR: om.ONNXGemmMetatype,
                NNCFNode.LAYER_ATTRIBUTES: ONNXLayerAttributes(
                    weight_attrs={1: {"shape": [5]}}, node_attrs={"transA": 0, "transB": 1}
                ),
            }
        ),
        target_point=ONNXTargetPoint(
            target_type=TargetType.OPERATION_WITH_WEIGHTS,
            target_node_name="gemm_with_weight_per_channel_0_port",
            port_id=1,
        ),
        per_channel=True,
        ref_reduction_shape=(0,),
    ),
    TestCase(
        nncf_node=NNCFNode(
            {
                NNCFNode.ID_NODE_ATTR: 0,
                NNCFNode.NODE_NAME_ATTR: "gemm_with_weight_per_channel_0_port",
                NNCFNode.METATYPE_ATTR: om.ONNXGemmMetatype,
                NNCFNode.LAYER_ATTRIBUTES: ONNXLayerAttributes(
                    weight_attrs={0: {"shape": [10, 10, 5]}}, node_attrs={"transA": 0, "transB": 1}
                ),
            }
        ),
        target_point=ONNXTargetPoint(
            target_type=TargetType.OPERATION_WITH_WEIGHTS,
            target_node_name="gemm_with_weight_per_channel_0_port",
            port_id=0,
        ),
        per_channel=True,
        ref_reduction_shape=(0, 1),
    ),
)


@pytest.mark.parametrize(
    "test_case",
    (test_cases),
    ids=[test_case.nncf_node.node_name for test_case in test_cases],
)
def test_get_reduction_shape(test_case):
    """Checks the correct return reduction shape in ONNXMinMaxAlgo.
    Edge cases:
    1) per-tensor.
    2) transpose axis of GEMM node.
    3) one dimensional weight tensor.
    """
    quantization_axis = get_quantization_axis(
        is_per_channel=test_case.per_channel, node=test_case.nncf_node, target_point=test_case.target_point
    )
    if quantization_axis is not None:  # Per-Channel
        reduction_shape = get_reduction_shape(
            test_case.nncf_node.layer_attributes.weight_attrs[test_case.target_point.port_id]["shape"],
            quantization_axis,
        )
        assert reduction_shape == test_case.ref_reduction_shape
    else:
        assert not test_case.per_channel
