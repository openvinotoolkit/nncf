# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from collections import Counter

import pytest

from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.onnx.graph.metatypes import onnx_metatypes as om
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from tests.onnx.models import LinearModel
from tests.onnx.models import ModelWithIntEdges
from tests.onnx.models import MultiInputOutputModel
from tests.onnx.models import OneDepthwiseConvolutionalModel
from tests.onnx.quantization.common import min_max_quantize_model

TEST_MODELS = [LinearModel, MultiInputOutputModel, ModelWithIntEdges, OneDepthwiseConvolutionalModel]
REF_METATYPES_COUNTERS = [
    [
        InputNoopMetatype,
        om.ONNXConvolutionMetatype,
        om.ONNXBatchNormMetatype,
        om.ONNXReluMetatype,
        om.ONNXGlobalAveragePoolMetatype,
        om.ONNXConvolutionMetatype,
        OutputNoopMetatype,
    ],
    [
        InputNoopMetatype,
        InputNoopMetatype,
        InputNoopMetatype,
        om.ONNXConcatMetatype,
        om.ONNXAddLayerMetatype,
        OutputNoopMetatype,
        OutputNoopMetatype,
    ],
    [InputNoopMetatype, om.ONNXConstantOfShapeMetatype, om.ONNXShapeMetatype, OutputNoopMetatype],
    [InputNoopMetatype, om.ONNXDepthwiseConvolutionMetatype, OutputNoopMetatype],
]
QUANTIZED_REF_METATYPES_COUNTERS = [
    REF_METATYPES_COUNTERS[0]
    + [
        om.ONNXQuantizeLinearMetatype,
        om.ONNXDequantizeLinearMetatype,
    ]
    * 5,
    REF_METATYPES_COUNTERS[1]
    + [
        om.ONNXQuantizeLinearMetatype,
        om.ONNXDequantizeLinearMetatype,
    ]
    * 2,
    REF_METATYPES_COUNTERS[2]
    + [
        om.ONNXQuantizeLinearMetatype,
        om.ONNXDequantizeLinearMetatype,
    ]
    * 0,
    REF_METATYPES_COUNTERS[3]
    + [
        om.ONNXQuantizeLinearMetatype,
        om.ONNXDequantizeLinearMetatype,
    ]
    * 2,
]


@pytest.mark.parametrize(
    ("model_creator_func, ref_metatypes, q_ref_metatypes"),
    zip(TEST_MODELS, REF_METATYPES_COUNTERS, QUANTIZED_REF_METATYPES_COUNTERS),
)
def test_mapping_onnx_metatypes(model_creator_func, ref_metatypes, q_ref_metatypes):
    def _check_metatypes(model, ref_metatypes):
        nncf_graph = GraphConverter.create_nncf_graph(model)
        actual_metatypes = [node.metatype for node in nncf_graph.get_all_nodes()]
        assert Counter(ref_metatypes) == Counter(actual_metatypes)

    model = model_creator_func().onnx_model
    q_model = min_max_quantize_model(model)
    _check_metatypes(model, ref_metatypes)
    _check_metatypes(q_model, q_ref_metatypes)


def test_all_onnx_metatypes_are_registered() -> None:
    """
    Checks that every metatype with op names defined in the onnx_metatypes module
    is registered in ONNX_OPERATION_METATYPES.
    """
    registered = set(om.ONNX_OPERATION_METATYPES.registry_dict.values())
    defined = [
        cls
        for _, cls in inspect.getmembers(om, inspect.isclass)
        if issubclass(cls, om.ONNXOpMetatype) and cls not in [om.ONNXOpMetatype, om.ONNXOpWithWeightsMetatype]
    ]
    assert defined, "No ONNX metatypes found in the module"
    not_registered = [cls.__name__ for cls in defined if cls not in registered]
    assert not not_registered, f"Metatypes are not registered in ONNX_OPERATION_METATYPES: {not_registered}"
