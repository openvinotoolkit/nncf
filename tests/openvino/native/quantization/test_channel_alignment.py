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

from enum import Enum
from typing import Type

import pytest

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.layout import OVLayoutElem
from nncf.openvino.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConstantMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.openvino.graph.transformations.commands import OVBiasInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import OVWeightUpdateCommand
from nncf.quantization.algorithms.channel_alignment.backend import LayoutDescriptor
from nncf.quantization.algorithms.channel_alignment.openvino_backend import OVChannelAlignmentAlgoBackend
from tests.cross_fw.test_templates.test_channel_alignment import TemplateTestChannelAlignment


class TestOVChannelAlignment(TemplateTestChannelAlignment):
    def get_backend_cls(self) -> Type[OVChannelAlignmentAlgoBackend]:
        return OVChannelAlignmentAlgoBackend

    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    def convert_conv_layer_attrs(self, layer_attributes):
        return OVLayerAttributes({}, layer_attributes)

    def get_conv_metatype(self):
        return OVConvolutionMetatype

    def get_add_metatype(self):
        return OVAddMetatype

    def get_add_layer_attrs(self):
        return OVLayerAttributes({1: 1}, {})

    def get_constant_metatype(self):
        return OVConstantMetatype

    def get_transformation_commands(self):
        return OVBiasInsertionCommand, OVBiasCorrectionCommand, OVWeightUpdateCommand

    def mock_command_creation_factory(self, mocker) -> None:
        mocker.patch("nncf.common.factory.CommandCreatorFactory.create", return_value=OVCommandCreator)

    class NodeType(Enum):
        CONVOLUTION = "CONVOLUTION"
        LINEAR = "LINEAR"

    @pytest.mark.parametrize(
        "weights_layout,node_type,ref_layout_desc",
        [
            (
                (OVLayoutElem.C_OUT, OVLayoutElem.C_IN, OVLayoutElem.SPATIAL, OVLayoutElem.SPATIAL),
                NodeType.CONVOLUTION,
                LayoutDescriptor(0, 1, 1),
            ),
            (
                (
                    OVLayoutElem.GROUPS,
                    OVLayoutElem.C_OUT,
                    OVLayoutElem.C_IN,
                    OVLayoutElem.SPATIAL,
                    OVLayoutElem.SPATIAL,
                ),
                NodeType.CONVOLUTION,
                LayoutDescriptor(0, 2, 1),
            ),
            ((OVLayoutElem.C_IN, OVLayoutElem.C_OUT), NodeType.LINEAR, LayoutDescriptor(1, 0, -1)),
            ((OVLayoutElem.C_IN,), NodeType.LINEAR, LayoutDescriptor(None, 0, -1)),
            (
                (
                    OVLayoutElem.SPATIAL,
                    OVLayoutElem.SPATIAL,
                    OVLayoutElem.SPATIAL,
                    OVLayoutElem.C_IN,
                    OVLayoutElem.C_OUT,
                ),
                NodeType.LINEAR,
                LayoutDescriptor(4, 3, -1),
            ),
        ],
    )
    def test_conv_params_dims(self, weights_layout, node_type, ref_layout_desc, mocker):
        base = "nncf.quantization.algorithms.channel_alignment.openvino_backend."
        conv_layout_path = base + "get_conv_weights_layout_from_node"
        linear_layout_path = base + "get_linear_weights_layout_from_node"

        if node_type == self.NodeType.CONVOLUTION:
            metatype = OVConvolutionMetatype

            mocker.patch(
                conv_layout_path,
                return_value=weights_layout,
            )
            mocker.patch(
                linear_layout_path,
                return_value=None,
            )
        else:
            metatype = OVMatMulMetatype
            mocker.patch(
                conv_layout_path,
                return_value=None,
            )
            mocker.patch(
                linear_layout_path,
                return_value=weights_layout,
            )
        node = NNCFNode({NNCFNode.METATYPE_ATTR: metatype})
        layout_descr = OVChannelAlignmentAlgoBackend.get_dims_descriptor(node)
        assert layout_descr == ref_layout_desc
