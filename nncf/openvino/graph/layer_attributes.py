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

from typing import Any, Dict, List, Optional

import openvino.runtime as ov

from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import LayoutElem
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.layer_attributes import WeightedLayerAttributes
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVDepthwiseConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionMetatype
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

    def get_backend_agnostic_attributes(self):
        return self._layer_attributes


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
    if ov_metatype in [
        OVConvolutionMetatype,
        OVDepthwiseConvolutionMetatype,
        OVGroupConvolutionMetatype,
        OVConvolutionBackpropDataMetatype,
        OVGroupConvolutionBackpropDataMetatype,
    ]:
        node_attrs = ov_node.get_attributes()
        kwargs = {
            "weight_requires_grad": False,
            "stride": tuple(node_attrs["strides"]),
            "dilations": node_attrs["dilations"],
            "transpose": ov_metatype in [OVConvolutionBackpropDataMetatype, OVGroupConvolutionBackpropDataMetatype],
            # TODO: ticket 114378: unify pad attribute
            "padding_values": tuple(node_attrs["pads_begin"] + node_attrs["pads_end"]),
        }

        weights_layout_map = {
            OVConvolutionMetatype: [LayoutElem.C_OUT, LayoutElem.C_IN],
            OVGroupConvolutionMetatype: [LayoutElem.GROUPS, LayoutElem.C_OUT, LayoutElem.C_IN],
            OVDepthwiseConvolutionMetatype: [LayoutElem.GROUPS, LayoutElem.C_OUT, LayoutElem.C_IN],
            OVConvolutionBackpropDataMetatype: [LayoutElem.C_IN, LayoutElem.C_OUT],
            OVGroupConvolutionBackpropDataMetatype: [LayoutElem.GROUPS, LayoutElem.C_IN, LayoutElem.C_OUT],
        }

        weights_layout = weights_layout_map[ov_metatype]
        weights_shape = attrs["shape"]
        kwargs.update(
            {
                "in_channels": weights_shape[weights_layout.index(LayoutElem.C_IN)],
                "out_channels": weights_shape[weights_layout.index(LayoutElem.C_OUT)],
                "kernel_size": tuple(weights_shape[len(weights_layout) :]),
                "groups": weights_shape[weights_layout.index(LayoutElem.GROUPS)]
                if LayoutElem.GROUPS in weights_layout
                else 1,
            }
        )
        kwargs.update({"weights_layout": tuple(weights_layout + len(kwargs["kernel_size"]) * [LayoutElem.SPATIAL])})

        return ConvolutionLayerAttributes(**kwargs)
    if ov_metatype == OVMatMulMetatype:
        weights_shape = attrs["shape"]

        weights_layout = [LayoutElem.SPATIAL] * (len(weights_shape) - 2)
        if len(weights_shape) > 1:
            transpose = attrs.get("transpose", False)
            if (transpose and port_id == 0) or (not transpose and port_id == 1):
                weights_layout += [LayoutElem.C_IN, LayoutElem.C_OUT]
            else:
                weights_layout += [LayoutElem.C_OUT, LayoutElem.C_IN]
        else:
            weights_layout += [LayoutElem.C_IN]

        kwargs = {
            "weight_requires_grad": False,
            "in_features": weights_shape[weights_layout.index(LayoutElem.C_IN)],
            "out_features": weights_shape[weights_layout.index(LayoutElem.C_OUT)]
            if LayoutElem.C_OUT in weights_layout
            else None,
            "with_bias": False,
            "weights_layout": weights_layout,
        }
        return LinearLayerAttributes(**kwargs)
    return GenericWeightedLayerAttributes(weight_requires_grad=False, weight_shape=attrs.get("shape", None))
