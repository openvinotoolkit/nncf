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

import pytest
from collections import Counter

from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConcatMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConstantMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMultiplyMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVReluMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVReshapeMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVSubtractMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVTransposeMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVParameterMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVResultMetatype
from nncf.experimental.openvino_native.graph.nncf_graph_builder import GraphConverter

from tests.openvino.native.models import ConvModel
from tests.openvino.native.models import LinearModel
from tests.openvino.native.models import WeightsModel

TEST_MODELS = [LinearModel, ConvModel]
REF_METATYPES_COUNTERS = [
    [OVParameterMetatype, OVConstantMetatype, OVReshapeMetatype,
     OVConstantMetatype, OVAddMetatype, OVConstantMetatype, OVMatMulMetatype,
     OVResultMetatype, OVResultMetatype],
    [OVParameterMetatype, OVParameterMetatype, OVConstantMetatype, OVMultiplyMetatype,
     OVConstantMetatype, OVAddMetatype, OVConstantMetatype, OVSubtractMetatype,
     OVConstantMetatype, OVConvolutionMetatype, OVReluMetatype, OVConcatMetatype,
     OVTransposeMetatype, OVConstantMetatype, OVResultMetatype]]


@pytest.mark.parametrize(("model_creator_func, ref_metatypes"),
                         zip(TEST_MODELS, REF_METATYPES_COUNTERS))
def test_mapping_openvino_metatypes(model_creator_func, ref_metatypes):
    model = model_creator_func().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    actual_metatypes = [node.metatype for node in nncf_graph.get_all_nodes()]
    assert Counter(ref_metatypes) == Counter(actual_metatypes)


REF_WEIGHTS_PORT_IDS = {
    'Conv': 1,
    'Conv_backprop': 1,
    'MatMul_1': 1,
    'MatMul_0': 0,
}


def test_determining_weights_port():
    model = WeightsModel().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    counter = 0
    for node in nncf_graph.get_all_nodes():
        if node.layer_attributes is not None:
            counter += 1
            assert node.layer_attributes.weight_port_id == REF_WEIGHTS_PORT_IDS[node.node_name]
    assert counter == len(REF_WEIGHTS_PORT_IDS)
