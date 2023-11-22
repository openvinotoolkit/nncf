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

from typing import Any, Dict, List, Optional, Tuple

import openvino.runtime as ov

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.layer_attributes import WeightedLayerAttributes
from nncf.openvino.graph.layout import OVLayoutElem
from nncf.openvino.graph.metatypes.groups import CONV_OPERATIONS
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVOpMetatype


class OVLayerAttributes(BaseLayerAttributes):
    """
    This class stores additional information about nodes that needs to be processed during compression.
    """

    def __init__(
        self,
        constant_attributes: Dict[int, Any],
        layer_attributes: Optional[BaseLayerAttributes] = None,
        inputs_attributes: Optional[Dict[Any, Any]] = None,
    ):
        """
        :param constant_attributes: Map of weights port ID to corresponding const attributes.
        :param layer_attributes: Map of weights port ID to corresponding common layer attributes.
        :param inputs_attributes: Activation attributes.
        """
        self._constant_attributes = constant_attributes
        self._layer_attributes = layer_attributes
        self._inputs_attributes = inputs_attributes

    @property
    def constant_attributes(self) -> Dict[int, Any]:
        return self._constant_attributes

    @property
    def layer_attributes(self) -> Optional[BaseLayerAttributes]:
        return self._layer_attributes

    @property
    def input_attributes(self) -> Optional[Dict[Any, Any]]:
        return self._inputs_attributes

    def get_const_port_ids(self) -> List[int]:
        """
        Returns indices of input ports corresponding to the constant nodes.

        :returns: List of input port indices with constants.
        """
        if self._constant_attributes is not None:
            return list(self._constant_attributes.keys())
        return []


def get_conv_weights_layout_from_node(node: NNCFNode) -> List[OVLayoutElem]:
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


def get_linear_weights_layout_from_node(node: NNCFNode) -> List[OVLayoutElem]:
    """
    Calculates weights layout for a target linear node.

    :param node: Target linear node.
    :return: Target linear Node weight layout.
    """
    layer_attributes = node.layer_attributes
    port_id = _get_constant_port_id_from_layer_attributes(layer_attributes)
    constant_layer_attrs = layer_attributes.constant_attributes[port_id]
    return get_linear_weights_layout(
        weights_shape=constant_layer_attrs["shape"],
        transpose=constant_layer_attrs["transpose"],
        port_id=port_id,
    )


def _get_constant_port_id_from_layer_attributes(layer_attributes: OVLayerAttributes) -> int:
    """
    Returns constant ports id for convolutional and linear ops layer attributes.

    :param layer_attributes: Target convolutional/linear layer op layer attributes.
    :return: Constant port id for the target convolutional/linear model.
    """
    port_ids = list(layer_attributes.constant_attributes.keys())
    assert len(port_ids) == 1
    return port_ids[0]


def get_conv_weights_layout(ov_metatype: OVOpMetatype, weights_shape: Tuple[int, ...]) -> List[OVLayoutElem]:
    """
    Calculates weights layout for a target convolution node.

    :param ov_metatype: Target convolution node OpenVINO metatype.
    :param weights_shape: Shape of the target convolution node weight.
    :return: Target convolution node weights layout.
    """
    weights_layout = ov_metatype.const_layout
    kernel_size = weights_shape[len(weights_layout) :]
    weights_layout += [OVLayoutElem.SPATIAL] * len(kernel_size)
    return tuple(weights_layout)


def get_linear_weights_layout(weights_shape: Tuple[int, ...], transpose: bool, port_id: int) -> List[OVLayoutElem]:
    """
    Calculates weights layout for a target linear node.

    :param weights_shape: Shape of the target linear node weight.
    :param port_id: Port id of the target liner node weights.
    :return: Target linear node weight layout.
    """
    weights_layout = [OVLayoutElem.SPATIAL] * (len(weights_shape) - 2)
    if len(weights_shape) > 1:
        if (transpose and port_id == 0) or (not transpose and port_id == 1):
            weights_layout += [OVLayoutElem.C_IN, OVLayoutElem.C_OUT]
        else:
            weights_layout += [OVLayoutElem.C_OUT, OVLayoutElem.C_IN]
    else:
        weights_layout += [OVLayoutElem.C_IN]
    return tuple(weights_layout)


def get_weighted_layer_attributes(
    ov_node: ov.Node, ov_metatype: OVOpMetatype, constant_attributes: Dict[int, Any]
) -> WeightedLayerAttributes:
    """
    Funciton retrieves common layer attributes from the given node.

    :param ov_node: TargetOpenvino graph node instance.
    :param ov_metatype: NNCF Openvino metatype of the given node.
    :param constant_attributes: Constant attributes collected for the given node.
    :return: Weighted layer attributes for the given node.
    """
    if len(constant_attributes) != 1:
        return None

    port_id, attrs = constant_attributes.copy().popitem()
    if ov_metatype in CONV_OPERATIONS:
        node_attrs = ov_node.get_attributes()
        kwargs = {
            "weight_requires_grad": False,
            "stride": tuple(node_attrs["strides"]),
            "dilations": node_attrs["dilations"],
            "transpose": ov_metatype in [OVConvolutionBackpropDataMetatype, OVGroupConvolutionBackpropDataMetatype],
            # TODO: ticket 114378: unify pad attribute
            "padding_values": tuple(node_attrs["pads_begin"] + node_attrs["pads_end"]),
        }
        weights_shape = attrs["shape"]
        weights_layout = get_conv_weights_layout(ov_metatype=ov_metatype, weights_shape=weights_shape)
        kwargs.update(
            {
                "in_channels": weights_shape[weights_layout.index(OVLayoutElem.C_IN)],
                "out_channels": weights_shape[weights_layout.index(OVLayoutElem.C_OUT)],
                "kernel_size": tuple(
                    dim for dim, elem in zip(weights_shape, weights_layout) if elem == OVLayoutElem.SPATIAL
                ),
                "groups": weights_shape[weights_layout.index(OVLayoutElem.GROUPS)]
                if OVLayoutElem.GROUPS in weights_layout
                else 1,
            }
        )

        return ConvolutionLayerAttributes(**kwargs)
    if ov_metatype == OVMatMulMetatype:
        weights_shape = attrs["shape"]
        weights_layout = get_linear_weights_layout(
            weights_shape=weights_shape, transpose=attrs["transpose"], port_id=port_id
        )

        kwargs = {
            "weight_requires_grad": False,
            "in_features": weights_shape[weights_layout.index(OVLayoutElem.C_IN)],
            "out_features": weights_shape[weights_layout.index(OVLayoutElem.C_OUT)]
            if OVLayoutElem.C_OUT in weights_layout
            else None,
            "with_bias": False,
        }
        return LinearLayerAttributes(**kwargs)
    return GenericWeightedLayerAttributes(weight_requires_grad=False, weight_shape=attrs.get("shape", None))
