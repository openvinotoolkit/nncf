# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXAddLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDepthwiseConvolutionMetatype
from nncf.onnx.graph.nncf_graph_builder import ONNXLayerAttributes
from nncf.quantization.algorithms.min_max.onnx_backend import ONNXMinMaxAlgoBackend
from tests.post_training.test_templates.models import NNCFGraphToTest
from tests.post_training.test_templates.models import NNCFGraphToTestDepthwiseConv
from tests.post_training.test_templates.models import NNCFGraphToTestSumAggregation
from tests.post_training.test_templates.test_quantizer_config import TemplateTestQuantizerConfig


class TestQuantizerConfig(TemplateTestQuantizerConfig):
    def get_algo_backend(self):
        return ONNXMinMaxAlgoBackend()

    @pytest.fixture
    def single_conv_nncf_graph(self) -> NNCFGraphToTest:
        conv_layer_attrs = ONNXLayerAttributes(weight_attrs={1: {"shape": [4, 4, 4, 4]}}, bias_attrs={})
        return NNCFGraphToTest(
            ONNXConvolutionMetatype,
            conv_layer_attrs,
            input_layer_attrs=ONNXLayerAttributes(),
            output_layer_attrs=ONNXLayerAttributes(),
            const_layer_attrs=ONNXLayerAttributes(),
        )

    @pytest.fixture
    def depthwise_conv_nncf_graph(self) -> NNCFGraphToTestDepthwiseConv:
        return NNCFGraphToTestDepthwiseConv(
            ONNXDepthwiseConvolutionMetatype,
            ONNXLayerAttributes(weight_attrs={1: {"shape": [4, 4, 4, 4]}}, bias_attrs={}),
            input_layer_attrs=ONNXLayerAttributes(),
            output_layer_attrs=ONNXLayerAttributes(),
            const_layer_attrs=ONNXLayerAttributes(),
        )

    @pytest.fixture
    def conv_sum_aggregation_nncf_graph(self) -> NNCFGraphToTestSumAggregation:
        conv_layer_attrs = ONNXLayerAttributes(weight_attrs={1: {"shape": [4, 4, 4, 4]}}, bias_attrs={})
        return NNCFGraphToTestSumAggregation(
            ONNXConvolutionMetatype,
            ONNXAddLayerMetatype,
            conv_layer_attrs,
            sum_layer_attrs=ONNXLayerAttributes(),
            input_layer_attrs=ONNXLayerAttributes(),
            output_layer_attrs=ONNXLayerAttributes(),
            const_layer_attrs=ONNXLayerAttributes(),
        )
