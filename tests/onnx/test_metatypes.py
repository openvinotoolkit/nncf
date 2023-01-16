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

from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDepthwiseConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXBatchNormMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXReluMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXGlobalAveragePoolMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConcatLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXAddLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConstantOfShapeMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXShapeMetatype

from tests.onnx.models import LinearModel
from tests.onnx.models import MultiInputOutputModel
from tests.onnx.models import ModelWithIntEdges
from tests.onnx.models import OneDepthwiseConvolutionalModel

TEST_MODELS = [LinearModel, MultiInputOutputModel, ModelWithIntEdges, OneDepthwiseConvolutionalModel]
REF_METATYPES_COUNTERS = [
    [InputNoopMetatype, ONNXConvolutionMetatype, ONNXBatchNormMetatype, ONNXReluMetatype, ONNXGlobalAveragePoolMetatype,
     ONNXConvolutionMetatype, OutputNoopMetatype],
    [InputNoopMetatype, InputNoopMetatype, InputNoopMetatype,
     ONNXConcatLayerMetatype, ONNXAddLayerMetatype, OutputNoopMetatype, OutputNoopMetatype],
    [InputNoopMetatype, ONNXConstantOfShapeMetatype, ONNXShapeMetatype, OutputNoopMetatype],
    [InputNoopMetatype, ONNXDepthwiseConvolutionMetatype, OutputNoopMetatype]]


@pytest.mark.parametrize(("model_creator_func, ref_metatypes"),
                         zip(TEST_MODELS, REF_METATYPES_COUNTERS))
def test_mapping_onnx_metatypes(model_creator_func, ref_metatypes):
    model = model_creator_func()
    nncf_graph = GraphConverter.create_nncf_graph(model.onnx_model)
    actual_metatypes = [node.metatype for node in nncf_graph.get_all_nodes()]
    assert Counter(ref_metatypes) == Counter(actual_metatypes)
