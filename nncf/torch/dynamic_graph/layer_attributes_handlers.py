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

from torch.nn import Conv1d
from torch.nn import Conv2d
from torch.nn import Conv3d
from torch.nn import ConvTranspose1d
from torch.nn import ConvTranspose2d
from torch.nn import ConvTranspose3d
from torch.nn import Linear
from torch.nn import Module as TorchModule

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import GetItemLayerAttributes
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.common.graph.layer_attributes import MultipleOutputLayerAttributes
from nncf.common.graph.layer_attributes import PadLayerAttributes
from nncf.common.graph.layer_attributes import PermuteLayerAttributes
from nncf.common.graph.layer_attributes import ReshapeLayerAttributes
from nncf.common.graph.layer_attributes import TransposeLayerAttributes
from nncf.common.graph.utils import get_concat_axis
from nncf.common.graph.utils import get_split_axis
from nncf.torch.graph.operator_metatypes import PTCatMetatype
from nncf.torch.graph.operator_metatypes import PTGroupNormMetatype
from nncf.torch.graph.operator_metatypes import PTPadMetatype
from nncf.torch.graph.operator_metatypes import PTReshapeMetatype
from nncf.torch.graph.operator_metatypes import PTSplitMetatype
from nncf.torch.layers import NNCF_MODULES_DICT

OP_NAMES_REQUIRING_MODULE_ATTRS = [v.op_func_name for v in NNCF_MODULES_DICT] + list(
    PTGroupNormMetatype.get_all_aliases()
)

TRANSPOSE_OP_NAMES = ["transpose", "transpose_"]
PERMUTE_OP_NAMES = ["permute"]
GETITEM_OP_NAMES = ["__getitem__"]
PAD_OP_NAMES = PTPadMetatype.get_all_aliases()
OP_NAMES_REQUIRING_ATTRS_FROM_ARGS_KWARGS = list(
    TRANSPOSE_OP_NAMES + PERMUTE_OP_NAMES + GETITEM_OP_NAMES + PAD_OP_NAMES
)


def get_layer_attributes_from_module(module: TorchModule, operator_name: str) -> BaseLayerAttributes:
    if operator_name == "group_norm":
        return GroupNormLayerAttributes(module.weight.requires_grad, module.num_channels, module.num_groups)
    # torch.nn.utils.weight_norm replaces weight with weight_g and weight_v
    is_weight_norm_applied = hasattr(module, "weight_g") and hasattr(module, "weight_v")
    weight_attr = "weight_g" if is_weight_norm_applied else "weight"
    if isinstance(module, (Conv1d, Conv2d, Conv3d)):
        return ConvolutionLayerAttributes(
            weight_requires_grad=getattr(module, weight_attr).requires_grad,
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            groups=module.groups,
            transpose=False,
            padding_values=module.padding,
        )
    if isinstance(module, (ConvTranspose1d, ConvTranspose2d, ConvTranspose3d)):
        return ConvolutionLayerAttributes(
            weight_requires_grad=getattr(module, weight_attr).requires_grad,
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            groups=module.groups,
            transpose=True,
            padding_values=module.padding,
        )
    if isinstance(module, Linear):
        return LinearLayerAttributes(
            weight_requires_grad=getattr(module, weight_attr).requires_grad,
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
        )

    if hasattr(module, "weight"):
        return GenericWeightedLayerAttributes(
            weight_requires_grad=getattr(module, weight_attr).requires_grad, weight_shape=module.weight.shape
        )

    return GenericWeightedLayerAttributes(weight_requires_grad=False, weight_shape=[1, 1])


def get_layer_attributes_from_args_and_kwargs(op_name: str, args, kwargs) -> BaseLayerAttributes:
    layer_attrs = None
    if op_name in TRANSPOSE_OP_NAMES:
        layer_attrs = _get_transpose_attrs_from_args_kwargs(args, kwargs)
    elif op_name in PERMUTE_OP_NAMES:
        layer_attrs = _get_permute_attrs_from_args_kwargs(args, kwargs)
    elif op_name in GETITEM_OP_NAMES:
        layer_attrs = _get_getitem_attrs_from_args_kwargs(args, kwargs)
    elif op_name in PAD_OP_NAMES:
        layer_attrs = _get_pad_attrs_from_args_kwargs(args, kwargs)
    return layer_attrs


def set_nodes_attributes_in_nncf_graph(graph: NNCFGraph) -> None:
    for node in graph.get_all_nodes():
        if node.metatype is PTCatMetatype:
            input_edges = graph.get_input_edges(node)
            output_edges = graph.get_output_edges(node)
            # Case of intermediate node
            if input_edges and output_edges:
                input_shapes = [edge.tensor_shape for edge in input_edges]
                output_shapes = [edge.tensor_shape for edge in output_edges]
                # Case node is stack
                if len(input_shapes[0]) != len(output_shapes[0]):
                    continue
                axis = get_concat_axis(input_shapes, output_shapes)
                layer_attributes = MultipleInputLayerAttributes(axis)
                node.layer_attributes = layer_attributes

        if node.metatype is PTReshapeMetatype:
            input_nodes = graph.get_input_edges(node)
            output_nodes = graph.get_output_edges(node)
            # In case ReshapeMetatype op is intermediate node
            if input_nodes and output_nodes:
                layer_attributes = ReshapeLayerAttributes(input_nodes[0].tensor_shape, output_nodes[0].tensor_shape)
                node.layer_attributes = layer_attributes

        if node.metatype is PTSplitMetatype:
            input_edges = graph.get_input_edges(node)
            output_edges = graph.get_output_edges(node)
            if input_edges and output_edges:
                input_shapes = [edge.tensor_shape for edge in input_edges]
                output_shapes = [edge.tensor_shape for edge in output_edges]
                axis = get_split_axis(input_shapes, output_shapes)
                chunks = len(output_edges)
                layer_attributes = MultipleOutputLayerAttributes(chunks, axis)
                node.layer_attributes = layer_attributes


def _get_transpose_attrs_from_args_kwargs(args, kwargs) -> TransposeLayerAttributes:
    return TransposeLayerAttributes(**_get_kwargs_shifted(["dim0", "dim1"], args, kwargs))


def _get_getitem_attrs_from_args_kwargs(args, kwargs):
    return GetItemLayerAttributes(key=args[1])


def _get_permute_attrs_from_args_kwargs(args, kwargs) -> PermuteLayerAttributes:
    arg_name = "dims"
    dims = kwargs[arg_name] if arg_name in kwargs else args[1:]
    return PermuteLayerAttributes(dims)


def _get_pad_attrs_from_args_kwargs(args, kwargs) -> PadLayerAttributes:
    mode = kwargs.get("mode", "constant" if len(args) < 3 else args[2])
    value = kwargs.get("value", 0 if len(args) < 4 else args[3])
    return PadLayerAttributes(mode, value)


def _get_kwargs_shifted(args_names, args, kwargs, shift=1):
    res_kwargs = {}
    for idx, arg_name in enumerate(args_names):
        res_kwargs[arg_name] = kwargs[arg_name] if arg_name in kwargs else args[idx + shift]
    return res_kwargs
