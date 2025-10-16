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

import pytest

from nncf.common.utils.backend import BackendType
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConstantMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVDepthwiseConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVEmbeddingMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMultiplyMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVReadValueMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVShapeOfMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVSoftmaxMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVSumMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVTransposeMetatype
from nncf.quantization.algorithms.min_max.openvino_backend import OVMinMaxAlgoBackend
from tests.cross_fw.test_templates.models import NNCFGraphConstantBranchWithWeightedNode
from tests.cross_fw.test_templates.models import NNCFGraphModelWithEmbeddingsConstantPath
from tests.cross_fw.test_templates.models import NNCFGraphModelWithEmbeddingsShapeOf
from tests.cross_fw.test_templates.models import NNCFGraphToTest
from tests.cross_fw.test_templates.models import NNCFGraphToTestDepthwiseConv
from tests.cross_fw.test_templates.models import NNCFGraphToTestSumAggregation
from tests.cross_fw.test_templates.models import NNCFGraphTransformer
from tests.cross_fw.test_templates.test_quantizer_config import TemplateTestQuantizerConfig


class TestQuantizerConfig(TemplateTestQuantizerConfig):
    def get_algo_backend(self):
        return OVMinMaxAlgoBackend()

    def get_backend_type(self):
        return BackendType.OPENVINO

    @pytest.fixture
    def single_conv_nncf_graph(self) -> NNCFGraphToTest:
        conv_layer_attrs = OVLayerAttributes({0: {"name": "dummy", "shape": (4, 4, 4, 4), "dtype": "f32"}})
        return NNCFGraphToTest(OVConvolutionMetatype, conv_layer_attrs)

    @pytest.fixture
    def depthwise_conv_nncf_graph(self):
        return NNCFGraphToTestDepthwiseConv(OVDepthwiseConvolutionMetatype, conv_layer_attrs=OVLayerAttributes({}))

    @pytest.fixture
    def conv_sum_aggregation_nncf_graph(self) -> NNCFGraphToTestSumAggregation:
        conv_layer_attrs = OVLayerAttributes({0: {"name": "dummy", "shape": (4, 4, 4, 4), "dtype": "f32"}})
        return NNCFGraphToTestSumAggregation(OVConvolutionMetatype, OVSumMetatype, conv_layer_attrs)

    @pytest.fixture
    def transformer_nncf_graph(self) -> NNCFGraphToTest:
        return NNCFGraphTransformer(
            matmul_metatype=OVMatMulMetatype,
            softmax_metatype=OVSoftmaxMetatype,
            mul_metatype=OVMultiplyMetatype,
            const_metatype=OVConstantMetatype,
            transpose_metatype=OVTransposeMetatype,
            matmul_layer_weighted_attrs=OVLayerAttributes({}),
        )

    @pytest.fixture
    def embedding_nncf_graph_shape_of(self) -> NNCFGraphToTest:
        return NNCFGraphModelWithEmbeddingsShapeOf(
            const_metatype=OVConstantMetatype,
            embedding_metatype=OVEmbeddingMetatype,
            conv_metatype=OVConvolutionMetatype,
            add_metatype=OVAddMetatype,
            shape_of_metatype=OVShapeOfMetatype,
            conv_layer_attrs=OVLayerAttributes({}),
            embedding_layer_attrs=OVLayerAttributes({}),
        )

    @pytest.fixture
    def embedding_nncf_graph_constant_path(self):
        return NNCFGraphModelWithEmbeddingsConstantPath(
            const_metatype=OVConstantMetatype,
            embedding_metatype=OVEmbeddingMetatype,
            conv_metatype=OVConvolutionMetatype,
            add_metatype=OVAddMetatype,
            conv_layer_attrs=OVLayerAttributes({}),
            embedding_layer_attrs=OVLayerAttributes({}),
        )

    @pytest.fixture
    def constant_branch_nncf_graph(self) -> NNCFGraphToTest:
        return NNCFGraphConstantBranchWithWeightedNode(
            const_metatype=OVConstantMetatype,
            conv_metatype=OVConvolutionMetatype,
            add_metatype=OVAddMetatype,
            conv_layer_attrs=OVLayerAttributes({}),
        )

    def test_self_attn_output_with_read_value(self):
        self.test_model_type_transformer_quantization_config(
            NNCFGraphTransformer(
                matmul_metatype=OVMatMulMetatype,
                softmax_metatype=OVSoftmaxMetatype,
                mul_metatype=OVMultiplyMetatype,
                const_metatype=OVConstantMetatype,
                transpose_metatype=OVReadValueMetatype,
                matmul_layer_weighted_attrs=OVLayerAttributes({}),
            ),
            dict(),
            self.get_ref_transformer_setup_state,
        )
