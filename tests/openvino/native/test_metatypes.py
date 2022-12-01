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

from collections import Counter

import pytest

from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConstantMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVReshapeMetatype
from nncf.experimental.openvino_native.graph.nncf_graph_builder import GraphConverter

from tests.openvino.native.models import LinearModel

REF_METATYPES_COUNTERS = [
    [InputNoopMetatype, OVConstantMetatype, OVReshapeMetatype,
     OVConstantMetatype, OVAddMetatype, OVConstantMetatype, OVMatMulMetatype,
     OutputNoopMetatype, OutputNoopMetatype]]


@pytest.mark.parametrize("ref_metatypes", REF_METATYPES_COUNTERS)
def test_mapping_onnx_metatypes(ref_metatypes):
    model = LinearModel().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    actual_metatypes = [node.metatype for node in nncf_graph.get_all_nodes()]
    assert Counter(ref_metatypes) == Counter(actual_metatypes)
