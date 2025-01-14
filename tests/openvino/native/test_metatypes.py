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

from collections import Counter

import pytest

import nncf.openvino.graph.metatypes.openvino_metatypes as ovm
from nncf.openvino.graph.metatypes.groups import OPERATIONS_WITH_WEIGHTS
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from tests.openvino.native.models import ConvModel
from tests.openvino.native.models import DepthwiseConv4DModel
from tests.openvino.native.models import LinearModel
from tests.openvino.native.models import PreluModel
from tests.openvino.native.models import WeightsModel

TEST_MODELS = [LinearModel, ConvModel, DepthwiseConv4DModel, PreluModel]
REF_METATYPES_COUNTERS = [
    [
        ovm.OVParameterMetatype,
        ovm.OVConstantMetatype,
        ovm.OVReshapeMetatype,
        ovm.OVConstantMetatype,
        ovm.OVAddMetatype,
        ovm.OVConstantMetatype,
        ovm.OVMatMulMetatype,
        ovm.OVResultMetatype,
        ovm.OVResultMetatype,
    ],
    [
        ovm.OVParameterMetatype,
        ovm.OVParameterMetatype,
        ovm.OVConstantMetatype,
        ovm.OVMultiplyMetatype,
        ovm.OVConstantMetatype,
        ovm.OVAddMetatype,
        ovm.OVConstantMetatype,
        ovm.OVSubtractMetatype,
        ovm.OVConstantMetatype,
        ovm.OVConvolutionMetatype,
        ovm.OVReluMetatype,
        ovm.OVConcatMetatype,
        ovm.OVTransposeMetatype,
        ovm.OVConstantMetatype,
        ovm.OVResultMetatype,
        ovm.OVAddMetatype,
        ovm.OVConstantMetatype,
    ],
    [
        ovm.OVParameterMetatype,
        ovm.OVConstantMetatype,
        ovm.OVDepthwiseConvolutionMetatype,
        ovm.OVConstantMetatype,
        ovm.OVAddMetatype,
        ovm.OVReluMetatype,
        ovm.OVResultMetatype,
    ],
    [ovm.OVPReluMetatype, ovm.OVResultMetatype, ovm.OVParameterMetatype, ovm.OVConstantMetatype],
]


@pytest.mark.parametrize(("model_creator_func, ref_metatypes"), zip(TEST_MODELS, REF_METATYPES_COUNTERS))
def test_mapping_openvino_metatypes(model_creator_func, ref_metatypes):
    model = model_creator_func().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    actual_metatypes = [node.metatype for node in nncf_graph.get_all_nodes()]
    assert Counter(ref_metatypes) == Counter(actual_metatypes)


REF_WEIGHTS_PORT_IDS = {
    "Conv": 1,
    "Conv_backprop": 1,
    "MatMul_1": 1,
    "MatMul_0": 0,
}


def test_determining_weights_port():
    model = WeightsModel().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    counter = 0
    for node in nncf_graph.get_all_nodes():
        if node.metatype not in OPERATIONS_WITH_WEIGHTS:
            continue
        if node.layer_attributes and node.layer_attributes.constant_attributes:
            counter += 1
            const_port_ids = node.layer_attributes.get_const_port_ids()
            assert len(const_port_ids) == 1
            const_port_id = const_port_ids[0]
            assert const_port_id == REF_WEIGHTS_PORT_IDS[node.node_name]
    assert counter == len(REF_WEIGHTS_PORT_IDS)
