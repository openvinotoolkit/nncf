from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypeVar

from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import MultipleWeightsLayerAttributes
from nncf.common.graph.layer_attributes import WeightedLayerAttributes
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVDepthwiseConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionMetatype


class OVConstantLayerAttributesContainer(BaseLayerAttributes):
    """
    This class stores mapping weights port indices to constant name and shape.
    """

    def __init__(self, const_attrs: Dict[int, Any], common_layer_attrs: Dict[int, BaseLayerAttributes]):
        """
        :param const_attrs: Map of weights port ID to corresponding const attributes.
        """
        self.const_attrs = const_attrs
        self.common_layer_attrs = common_layer_attrs

    def get_const_port_ids(self) -> List[int]:
        """
        Returns indices of input ports corresponding to the constant nodes.

        :returns: List of input port indices with constants.
        """
        return list(self.const_attrs.keys())


def get_weighted_layer_attributes(ov_node, ov_metatype, constant_attributes: Dict[str, Any]):
    retval = {}
    for port_id, attrs in constant_attributes.items():
        if ov_metatype in [OVConvolutionMetatype, OVDepthwiseConvolutionMetatype, OVGroupConvolutionMetatype]:
            node_attrs = ov_node.get_attributes()
            kwargs = {
                "weight_requires_grad": False,
                "stride": tuple(node_attrs["strides"]),
                "dilations": node_attrs["dilations"],
                "transpose": attrs.get("transpose", False),
                # TODO: Unify pad attribute
                "padding_values": tuple(node_attrs["pads_begin"] + node_attrs["pads_end"]),
            }

            const_shape = attrs["shape"]
            if ov_metatype == OVConvolutionMetatype:
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

            common_layer_attr = ConvolutionLayerAttributes(**kwargs)
        else:
            common_layer_attr = GenericWeightedLayerAttributes(
                weight_requires_grad=False, weight_shape=attrs.get("shape", None)
            )
        retval[port_id] = common_layer_attr
    return retval
