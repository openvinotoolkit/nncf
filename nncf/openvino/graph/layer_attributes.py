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

from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import WeightedLayerAttributes
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVDepthwiseConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionMetatype


class OVLayerAttributes(BaseLayerAttributes):
    """
    This class stores additional information about nodes that needs to be processed during compression.
    """

    def __init__(
        self,
        const_attrs: Dict[int, Any],
        common_layer_attrs: Optional[Dict[int, BaseLayerAttributes]] = None,
        act_attrs: Optional[Dict[Any, Any]] = None,
    ):
        """
        :param const_attrs: Map of weights port ID to corresponding const attributes.
        :param common_layer_attrs: Map of weights port ID to corresponding common layer attributes.
        :param act_attrs: Activation attributes.
        """
        self._const_attrs = const_attrs
        self._common_layer_attrs = common_layer_attrs
        self._act_attrs = act_attrs

    @property
    def const_attrs(self) -> Dict[int, Any]:
        return self._const_attrs

    @property
    def common_layer_attrs(self) -> Optional[Dict[int, BaseLayerAttributes]]:
        return self._common_layer_attrs

    @property
    def act_attrs(self) -> Optional[Dict[Any, Any]]:
        return self._act_attrs

    def get_const_port_ids(self) -> List[int]:
        """
        Returns indices of input ports corresponding to the constant nodes.

        :returns: List of input port indices with constants.
        """
        if self._const_attrs is not None:
            return list(self._const_attrs.keys())
        return []


def get_weighted_layer_attributes(ov_node, ov_metatype, constant_attributes: Dict[str, Any]) -> WeightedLayerAttributes:
    """
    Funciton retrieves common layer attributes from the given node.

    :param ov_node: TargetOpenvino graph node instance.
    :param ov_metatype: NNCF Openvino metatype of the given node.
    :param constant_attributes: Constant attributes collected for the given node.
    :return: Weighted layer attributes for the given node.
    """
    retval = {}
    for port_id, attrs in constant_attributes.items():
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

            const_shape = attrs["shape"]
            if ov_metatype in [OVConvolutionMetatype, OVConvolutionBackpropDataMetatype]:
                kwargs.update(
                    {
                        "in_channels": const_shape[1],
                        "out_channels": const_shape[0],
                        "kernel_size": tuple(const_shape[2:]),
                        "groups": 1,
                    }
                )
            else:
                kwargs.update(
                    {
                        "in_channels": const_shape[2],
                        "out_channels": const_shape[1],
                        "kernel_size": tuple(const_shape[3:]),
                        "groups": const_shape[0],
                    }
                )
            if kwargs["transpose"]:
                kwargs["in_channels"], kwargs["out_channels"] = kwargs["out_channels"], kwargs["in_channels"]

            common_layer_attr = ConvolutionLayerAttributes(**kwargs)
        else:
            common_layer_attr = GenericWeightedLayerAttributes(
                weight_requires_grad=False, weight_shape=attrs.get("shape", None)
            )
        retval[port_id] = common_layer_attr
    return retval
