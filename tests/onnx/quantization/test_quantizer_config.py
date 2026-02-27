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

import pytest

from nncf.common.utils.backend import BackendType
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXAddLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConstantMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDepthwiseConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXEmbeddingMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXMatMulMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXMulLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXShapeMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXSoftmaxMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXSplitMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXTransposeMetatype
from nncf.onnx.graph.nncf_graph_builder import ONNXLayerAttributes
from nncf.quantization.algorithms.min_max.onnx_backend import ONNXMinMaxAlgoBackend
from tests.cross_fw.test_templates.models import NNCFGraphArithmeticDegree2
from tests.cross_fw.test_templates.models import NNCFGraphConstantBranchWithWeightedNode
from tests.cross_fw.test_templates.models import NNCFGraphModelWithEmbeddingsConstantPath
from tests.cross_fw.test_templates.models import NNCFGraphModelWithEmbeddingsShapeOf
from tests.cross_fw.test_templates.models import NNCFGraphToTest
from tests.cross_fw.test_templates.models import NNCFGraphToTestDepthwiseConv
from tests.cross_fw.test_templates.models import NNCFGraphToTestSumAggregation
from tests.cross_fw.test_templates.models import NNCFGraphTransformer
from tests.cross_fw.test_templates.models import NNCFSplitGraphTransformer
from tests.cross_fw.test_templates.test_quantizer_config import TemplateTestQuantizerConfig


class TestQuantizerConfig(TemplateTestQuantizerConfig):
    def get_algo_backend(self):
        return ONNXMinMaxAlgoBackend()

    def get_backend_type(self):
        return BackendType.ONNX

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
    def single_conv_arithmetic_degree2_nncf_graph(self) -> NNCFGraphArithmeticDegree2:
        conv_layer_attrs = ONNXLayerAttributes(weight_attrs={1: {"shape": [4, 4, 4, 4]}}, bias_attrs={})
        return NNCFGraphArithmeticDegree2(
            ONNXConvolutionMetatype,
            ONNXAddLayerMetatype,
            conv_layer_attrs,
            arithmetic_layer_attrs=ONNXLayerAttributes(),
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

    @pytest.fixture
    def transformer_nncf_graph(self) -> NNCFGraphToTest:
        return NNCFGraphTransformer(
            matmul_metatype=ONNXMatMulMetatype,
            softmax_metatype=ONNXSoftmaxMetatype,
            mul_metatype=ONNXMulLayerMetatype,
            const_metatype=ONNXConstantMetatype,
            transpose_metatype=ONNXTransposeMetatype,
            matmul_layer_weighted_attrs=ONNXLayerAttributes({"name": "edge_name", "shape": (1, 1, 1, 1)}),
            matmul_layer_non_weighted_attrs=ONNXLayerAttributes(),
            default_layer_attrs=ONNXLayerAttributes(),
        )

    @pytest.fixture
    def split_transformer_nncf_graph(self) -> NNCFSplitGraphTransformer:
        conv_layer_attrs = ONNXLayerAttributes(weight_attrs={1: {"shape": [4, 4, 4, 4]}}, bias_attrs={})
        return NNCFSplitGraphTransformer(
            matmul_metatype=ONNXMatMulMetatype,
            conv_metatype=ONNXConvolutionMetatype,
            split_metatype=ONNXSplitMetatype,
            softmax_metatype=ONNXSoftmaxMetatype,
            const_metatype=ONNXConstantMetatype,
            mul_metatype=ONNXMulLayerMetatype,
            conv_layer_weighted_attrs=conv_layer_attrs,
            matmul_layer_non_weighted_attrs=ONNXLayerAttributes(),
            default_layer_attrs=ONNXLayerAttributes(),
        )

    @pytest.fixture
    def embedding_nncf_graph_shape_of(self) -> NNCFGraphToTest:
        return NNCFGraphModelWithEmbeddingsShapeOf(
            const_metatype=ONNXConstantMetatype,
            embedding_metatype=ONNXEmbeddingMetatype,
            conv_metatype=ONNXConvolutionMetatype,
            add_metatype=ONNXAddLayerMetatype,
            shape_of_metatype=ONNXShapeMetatype,
            conv_layer_attrs=ONNXLayerAttributes(weight_attrs={1: {"shape": [1, 1, 1, 1]}}, bias_attrs={}),
            embedding_layer_attrs=ONNXLayerAttributes(weight_attrs={1: {"shape": [1, 1, 1, 1]}}, bias_attrs={}),
            default_layer_attrs=ONNXLayerAttributes(),
        )

    @pytest.fixture
    def embedding_nncf_graph_constant_path(self) -> NNCFGraphToTest:
        return NNCFGraphModelWithEmbeddingsConstantPath(
            const_metatype=ONNXConstantMetatype,
            embedding_metatype=ONNXEmbeddingMetatype,
            conv_metatype=ONNXConvolutionMetatype,
            add_metatype=ONNXAddLayerMetatype,
            conv_layer_attrs=ONNXLayerAttributes(weight_attrs={1: {"shape": [1, 1, 1, 1]}}, bias_attrs={}),
            embedding_layer_attrs=ONNXLayerAttributes(weight_attrs={1: {"shape": [1, 1, 1, 1]}}, bias_attrs={}),
            default_layer_attrs=ONNXLayerAttributes(),
        )

    @pytest.fixture
    def constant_branch_nncf_graph(self) -> NNCFGraphToTest:
        return NNCFGraphConstantBranchWithWeightedNode(
            const_metatype=ONNXConstantMetatype,
            conv_metatype=ONNXConvolutionMetatype,
            add_metatype=ONNXAddLayerMetatype,
            conv_layer_attrs=ONNXLayerAttributes(weight_attrs={1: {"shape": [1, 1, 1, 1]}}, bias_attrs={}),
            default_layer_attrs=ONNXLayerAttributes(),
        )
