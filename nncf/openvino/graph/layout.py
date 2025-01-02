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
from typing import Tuple

from nncf.common.graph.graph import NNCFNode
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVDepthwiseConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVOpMetatype


class OVLayoutElem(Enum):
    """
    Layout elements descriptor for convolutional and linear openvino layers:
        C_IN: Input channels dimension.
        C_OUT: Output channels dimension.
        SPATIAL: Spatial dimension.
        GROUPS: Groups dimension.
    """

    C_IN = "channels_in"
    C_OUT = "channels_out"
    SPATIAL = "spatial"
    GROUPS = "groups"


_CONV_BASE_CONST_LAYOUT = {
    OVConvolutionMetatype: (OVLayoutElem.C_OUT, OVLayoutElem.C_IN),
    OVConvolutionBackpropDataMetatype: (OVLayoutElem.C_IN, OVLayoutElem.C_OUT),
    OVDepthwiseConvolutionMetatype: (OVLayoutElem.GROUPS, OVLayoutElem.C_OUT, OVLayoutElem.C_IN),
    OVGroupConvolutionMetatype: (OVLayoutElem.GROUPS, OVLayoutElem.C_OUT, OVLayoutElem.C_IN),
    OVGroupConvolutionBackpropDataMetatype: (OVLayoutElem.GROUPS, OVLayoutElem.C_IN, OVLayoutElem.C_OUT),
}


def get_conv_weights_layout_from_node(node: NNCFNode) -> Tuple[OVLayoutElem]:
    """
    Calculates weights layout for a target convolution node.

    :param node: Target convolution node.
    :return: Target convolution Node weights layout.
    """
    layer_attributes = node.layer_attributes
    port_id = _get_constant_port_id_from_layer_attributes(layer_attributes)
    return get_conv_weights_layout(
        ov_metatype=node.metatype, weights_shape=layer_attributes.constant_attributes[port_id]["shape"]
    )


def get_linear_weights_layout_from_node(node: NNCFNode) -> Tuple[OVLayoutElem]:
    """
    Calculates weights layout for a target linear node.

    :param node: Target linear node.
    :return: Target linear Node weight layout.
    """
    layer_attributes = node.layer_attributes
    port_id = _get_constant_port_id_from_layer_attributes(layer_attributes)
    constant_layer_attrs = layer_attributes.constant_attributes[port_id]
    return get_linear_input_layout(
        input_shape=constant_layer_attrs["shape"],
        transpose=constant_layer_attrs["transpose"],
        port_id=port_id,
    )


def get_linear_activations_layout_from_node(
    node: NNCFNode, port_id: int, input_shape: Tuple[int]
) -> Tuple[OVLayoutElem]:
    """
    Calculates activations layout for a target linear node.

    :param node: Target linear node.
    :param port_id: Target input port ID.
    :param input_shape: Shape of the input.
    :return: Target linear Node weight layout.
    """
    act_layer_attrs = node.layer_attributes.input_attributes
    return get_linear_input_layout(
        input_shape=input_shape,
        transpose=act_layer_attrs["transpose"],
        port_id=port_id,
    )


def get_conv_weights_layout(ov_metatype: OVOpMetatype, weights_shape: Tuple[int, ...]) -> Tuple[OVLayoutElem]:
    """
    Calculates weights layout for a target convolution node.

    :param ov_metatype: Target convolution node OpenVINO metatype.
    :param weights_shape: Shape of the target convolution node weight.
    :return: Target convolution node weights layout.
    """
    base_layout = _CONV_BASE_CONST_LAYOUT[ov_metatype]
    kernel_size = weights_shape[len(base_layout) :]
    weights_layout = list(base_layout) + [OVLayoutElem.SPATIAL] * len(kernel_size)
    return tuple(weights_layout)


def get_linear_input_layout(input_shape: Tuple[int, ...], transpose: bool, port_id: int) -> Tuple[OVLayoutElem]:
    """
    Calculates input layout for a target linear node.

    :param input_shape: Shape of the target linear node input.
    :param port_id: Port id of the target linear node input.
    :return: Target linear node input layout.
    """
    input_layout = [OVLayoutElem.SPATIAL] * (len(input_shape) - 2)
    if len(input_shape) > 1:
        if (transpose and port_id == 0) or (not transpose and port_id == 1):
            input_layout += [OVLayoutElem.C_IN, OVLayoutElem.C_OUT]
        else:
            input_layout += [OVLayoutElem.C_OUT, OVLayoutElem.C_IN]
    else:
        input_layout += [OVLayoutElem.C_IN]
    return tuple(input_layout)


def _get_constant_port_id_from_layer_attributes(layer_attributes: OVLayerAttributes) -> int:
    """
    Returns constant ports id for convolutional and linear ops layer attributes.

    :param layer_attributes: Target convolutional/linear layer op layer attributes.
    :return: Constant port id for the target convolutional/linear model.
    """
    port_ids = list(layer_attributes.constant_attributes.keys())
    assert len(port_ids) == 1
    return port_ids[0]
