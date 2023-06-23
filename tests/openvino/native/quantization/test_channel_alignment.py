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

from typing import Type

import pytest

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.openvino.graph.layer_attributes import OVConstantLayerAttributesContainer
from nncf.openvino.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConstantMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import OVWeightUpdateCommand
from nncf.quantization.algorithms.channel_alignment.openvino_backend import OVChannelAlignmentAlgoBackend
from tests.post_training.test_templates.test_channel_alignment import TemplateTestChannelAlignment


def create_nncf_graph_for_ca_algo():
    NNCFGraph()


NNCF_GRAPH_FOR_CA = None


class TestOVChannelAlignment(TemplateTestChannelAlignment):
    def get_backend_cls(self) -> Type[OVChannelAlignmentAlgoBackend]:
        return OVChannelAlignmentAlgoBackend

    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    def convert_conv_layer_attrs(self, layer_attributes):
        return OVConstantLayerAttributesContainer({}, {1: layer_attributes})

    def get_conv_metatype(self):
        return OVConvolutionMetatype

    def get_add_metatype(self):
        return OVAddMetatype

    def get_add_layer_attrs(self):
        return OVConstantLayerAttributesContainer({1: 1}, {})

    def get_constant_metatype(self):
        return OVConstantMetatype

    def get_transformation_commands(self):
        return OVBiasCorrectionCommand, OVWeightUpdateCommand

    @pytest.fixture(scope="session")
    def test_params(self):
        return {"test_get_node_pairs": {"NNCFGraph": {{"bad": None, "good": None}}}}
