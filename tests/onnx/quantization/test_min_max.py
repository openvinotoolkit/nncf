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
from nncf.onnx.graph.nncf_graph_builder import ONNXLayerAttributes
from nncf.quantization.algorithms.min_max.onnx_backend import ONNXMinMaxAlgoBackend

# pylint: disable=protected-access


@dataclass
class TestCase:
    nncf_node: NNCFNode
    weight_port_id: int
    per_channel: bool
    ref_reduction_shape: List[int]


unused_attrs = {"node_id": 0}

test_cases = (
    TestCase(
        nncf_node=NNCFNode(
            **unused_attrs,
            node_name="conv_with_weight_per_tensor",
            data={
                "layer_attributes": ONNXLayerAttributes(weight_attrs={1: {"shape": [3, 5, 8]}}),
                "metatype": om.ONNXConvolutionMetatype,
            },
        ),
        weight_port_id=1,
        per_channel=False,
        ref_reduction_shape=None,
    ),
    TestCase(
        nncf_node=NNCFNode(
            **unused_attrs,
            node_name="conv_with_weight_per_channel",
            data={
                "layer_attributes": ONNXLayerAttributes(weight_attrs={1: {"shape": [3, 5, 8]}}),
                "metatype": om.ONNXConvolutionMetatype,
            },
        ),
        weight_port_id=1,
        per_channel=True,
        ref_reduction_shape=(1, 2),
    ),
    TestCase(
        nncf_node=NNCFNode(
            **unused_attrs,
            node_name="gemm_with_weight_per_tensor",
            data={
                "layer_attributes": ONNXLayerAttributes(weight_attrs={1: {"shape": [5, 8]}}),
                "metatype": om.ONNXGemmMetatype,
            },
        ),
        weight_port_id=1,
        per_channel=False,
        ref_reduction_shape=None,
    ),
    TestCase(
        nncf_node=NNCFNode(
            **unused_attrs,
            node_name="gemm_with_weight_per_channel",
            data={
                "layer_attributes": ONNXLayerAttributes(weight_attrs={1: {"shape": [5, 8]}}),
                "metatype": om.ONNXGemmMetatype,
            },
        ),
        weight_port_id=1,
        per_channel=True,
        ref_reduction_shape=(0,),
    ),
    TestCase(
        nncf_node=NNCFNode(
            **unused_attrs,
            node_name="gemm_with_weight_per_channel_extra_attrs",
            data={
                "layer_attributes": ONNXLayerAttributes(
                    weight_attrs={1: {"shape": [5, 8]}}, node_attrs={"transA": 0, "transB": 0}
                ),
                "metatype": om.ONNXGemmMetatype,
            },
        ),
        weight_port_id=1,
        per_channel=True,
        ref_reduction_shape=(0,),
    ),
    TestCase(
        nncf_node=NNCFNode(
            **unused_attrs,
            node_name="gemm_with_weight_per_channel_extra_attrs",
            data={
                "layer_attributes": ONNXLayerAttributes(
                    weight_attrs={1: {"shape": [5, 8]}}, node_attrs={"transA": 1, "transB": 0}
                ),
                "metatype": om.ONNXGemmMetatype,
            },
        ),
        weight_port_id=1,
        per_channel=True,
        ref_reduction_shape=(0,),
    ),
    TestCase(
        nncf_node=NNCFNode(
            **unused_attrs,
            node_name="gemm_with_weight_per_channel_transpose",
            data={
                "layer_attributes": ONNXLayerAttributes(
                    weight_attrs={1: {"shape": [5, 8]}}, node_attrs={"transA": 0, "transB": 1}
                ),
                "metatype": om.ONNXGemmMetatype,
            },
        ),
        weight_port_id=1,
        per_channel=True,
        ref_reduction_shape=(1,),
    ),
    TestCase(
        nncf_node=NNCFNode(
            **unused_attrs,
            node_name="gemm_with_weight_per_channel_transpose_one_dim",
            data={
                "layer_attributes": ONNXLayerAttributes(
                    weight_attrs={1: {"shape": [5]}}, node_attrs={"transA": 0, "transB": 1}
                ),
                "metatype": om.ONNXGemmMetatype,
            },
        ),
        weight_port_id=1,
        per_channel=True,
        ref_reduction_shape=(0,),
    ),
    TestCase(
        nncf_node=NNCFNode(
            **unused_attrs,
            node_name="gemm_with_weight_per_channel_0_port",
            data={
                "layer_attributes": ONNXLayerAttributes(
                    weight_attrs={0: {"shape": [10, 10, 5]}}, node_attrs={"transA": 0, "transB": 1}
                ),
                "metatype": om.ONNXGemmMetatype,
            },
        ),
        weight_port_id=0,
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
    reduction_shape = ONNXMinMaxAlgoBackend()._get_reduction_shape_for_weight(
        test_case.nncf_node, test_case.weight_port_id, test_case.per_channel
    )
    assert reduction_shape == test_case.ref_reduction_shape
