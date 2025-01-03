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
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConstantMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVDepthwiseConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMultiplyMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVSoftmaxMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVSumMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVTransposeMetatype
from nncf.quantization.algorithms.min_max.openvino_backend import OVMinMaxAlgoBackend
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
