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

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConstantMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import OVWeightUpdateCommand
from nncf.quantization.algorithms.channel_alignment.backend import LayoutDescriptor
from nncf.quantization.algorithms.channel_alignment.openvino_backend import OVChannelAlignmentAlgoBackend
from tests.post_training.test_templates.test_channel_alignment import TemplateTestChannelAlignment


def _get_nncf_node(metatype, layer_attrs):
    return NNCFNode(0, "test", {NNCFGraph.METATYPE_ATTR: metatype, NNCFGraph.LAYER_ATTRIBUTES: layer_attrs})


class TestOVChannelAlignment(TemplateTestChannelAlignment):
    def get_backend_cls(self) -> Type[OVChannelAlignmentAlgoBackend]:
        return OVChannelAlignmentAlgoBackend

    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    def convert_conv_layer_attrs(self, layer_attributes):
        return OVLayerAttributes({}, {1: layer_attributes})

    def get_conv_metatype(self):
        return OVConvolutionMetatype

    def get_add_metatype(self):
        return OVAddMetatype

    def get_add_layer_attrs(self):
        return OVLayerAttributes({1: 1}, {})

    def get_constant_metatype(self):
        return OVConstantMetatype

    def get_transformation_commands(self):
        return OVBiasCorrectionCommand, OVWeightUpdateCommand

    def mock_command_creation_factory(self, mocker) -> None:
        mocker.patch("nncf.common.factory.CommandCreatorFactory.create", return_value=OVCommandCreator)

    @pytest.mark.parametrize("transpose", [False, True])
    @pytest.mark.parametrize("shape", [[3, 4], [1, 2, 3, 4]])
    @pytest.mark.parametrize("port_id", [-1, -2])
    def test_get_dims_descriptor_matmul(self, transpose, shape, port_id):
        _port_id = len(shape) + port_id
        node = _get_nncf_node(OVMatMulMetatype, OVLayerAttributes({_port_id: {"transpose": transpose, "shape": shape}}))
        dims_descr = OVChannelAlignmentAlgoBackend.get_dims_descriptor(node)

        in_dims, out_dims = (0, 1) if port_id == -1 else (1, 0)
        if len(shape) > 2:
            in_dims += 2
            out_dims += 2
        if transpose:
            in_dims, out_dims = out_dims, in_dims

        assert dims_descr.conv_weight_in_channels_dim == in_dims
        assert dims_descr.conv_weight_out_channels_dim == out_dims
        assert dims_descr.bias_channels_dim == OVMatMulMetatype.output_channel_axis

    def test_get_dims_descriptor_mm_no_layer_attrs(self):
        node = _get_nncf_node(OVMatMulMetatype, None)
        with pytest.raises(RuntimeError):
            OVChannelAlignmentAlgoBackend.get_dims_descriptor(node)

    @pytest.mark.parametrize(
        "metatype,ref_desc",
        [
            (OVConvolutionMetatype, LayoutDescriptor(0, 1, 1)),
            (OVGroupConvolutionMetatype, LayoutDescriptor(0, 2, 1)),
            (OVGroupConvolutionMetatype, LayoutDescriptor(0, 2, 1)),
        ],
    )
    def test_get_dims_descriptor_convs(self, metatype, ref_desc):
        node = _get_nncf_node(metatype, None)
        dims_descr = OVChannelAlignmentAlgoBackend.get_dims_descriptor(node)
        assert dims_descr.__dict__ == ref_desc.__dict__
