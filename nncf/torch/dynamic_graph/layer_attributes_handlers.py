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

from typing import Any, Dict, List, Tuple, Union

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConstantLayerAttributes
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
from nncf.common.graph.operator_metatypes import ConstNoopMetatype
from nncf.common.graph.operator_metatypes import get_all_aliases
from nncf.common.graph.utils import get_split_axis
from nncf.torch.dynamic_graph.trace_tensor import TracedParameter

TRANSPOSE_OP_NAMES = ["transpose", "transpose_"]
PERMUTE_OP_NAMES = ["permute"]
GETITEM_OP_NAMES = ["__getitem__"]
CONV_OP_NAMES = get_all_aliases(om.PTConv1dMetatype, om.PTConv2dMetatype, om.PTConv3dMetatype)
CONV_TRANSPOSE_OP_NAMES = get_all_aliases(
    om.PTConvTranspose1dMetatype, om.PTConvTranspose2dMetatype, om.PTConvTranspose3dMetatype
)
LINEAR_OP_NAMES = get_all_aliases(om.PTLinearMetatype)
BATCHNORM_OP_NAMES = get_all_aliases(om.PTBatchNormMetatype)
EMBEDDING_OP_NAMES = get_all_aliases(om.PTEmbeddingMetatype, om.PTEmbeddingBagMetatype)
GROUP_NORM_OP_NAMES = get_all_aliases(om.PTGroupNormMetatype)
LAYER_NORM_OP_NAMES = get_all_aliases(om.PTLayerNormMetatype)
PAD_OP_NAMES = om.PTPadMetatype.get_all_aliases()
CONCAT_OP_NAMES = om.PTCatMetatype.get_all_aliases()
CONST_OP_NAMES = ConstNoopMetatype.get_all_aliases()


def get_layer_attributes_from_args_and_kwargs(op_name: str, args, kwargs) -> BaseLayerAttributes:
    layer_attrs = None
    if op_name in CONV_OP_NAMES:
        layer_attrs = _get_conv_attrs_from_args_kwargs(args, kwargs)
    elif op_name in CONV_TRANSPOSE_OP_NAMES:
        layer_attrs = _get_conv_transpose_attrs_from_args_kwargs(args, kwargs)
    elif op_name in LINEAR_OP_NAMES:
        layer_attrs = _get_linear_attrs_from_args_kwargs(args, kwargs)
    elif op_name in GROUP_NORM_OP_NAMES:
        layer_attrs = _get_group_norm_attrs_from_args_kwargs(args, kwargs)
    elif op_name in BATCHNORM_OP_NAMES:
        layer_attrs = _get_batchnorm_attrs_from_args_kwargs(args, kwargs)
    elif op_name in LAYER_NORM_OP_NAMES:
        layer_attrs = _get_layer_norm_attrs_from_args_kwargs(args, kwargs)
    elif op_name in EMBEDDING_OP_NAMES:
        layer_attrs = _get_embedding_attrs_from_args_kwargs(args, kwargs)
    elif op_name in TRANSPOSE_OP_NAMES:
        layer_attrs = _get_transpose_attrs_from_args_kwargs(args, kwargs)
    elif op_name in PERMUTE_OP_NAMES:
        layer_attrs = _get_permute_attrs_from_args_kwargs(args, kwargs)
    elif op_name in GETITEM_OP_NAMES:
        layer_attrs = _get_getitem_attrs_from_args_kwargs(args, kwargs)
    elif op_name in PAD_OP_NAMES:
        layer_attrs = _get_pad_attrs_from_args_kwargs(args, kwargs)
    elif op_name in CONCAT_OP_NAMES:
        layer_attrs = _get_concat_attrs_from_args_kwargs(args, kwargs)
    elif op_name in CONST_OP_NAMES:
        layer_attrs = _get_const_attrs_from_args_kwargs(args, kwargs)
    return layer_attrs


def set_nodes_attributes_in_nncf_graph(graph: NNCFGraph) -> None:
    for node in graph.get_all_nodes():
        if node.metatype in [om.PTReshapeMetatype, om.PTSqueezeMetatype]:
            input_nodes = graph.get_input_edges(node)
            output_nodes = graph.get_output_edges(node)
            # In case ReshapeMetatype op is intermediate node
            if input_nodes and output_nodes:
                layer_attributes = ReshapeLayerAttributes(input_nodes[0].tensor_shape, output_nodes[0].tensor_shape)
                node.layer_attributes = layer_attributes

        if node.metatype is om.PTSplitMetatype:
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


def _get_concat_attrs_from_args_kwargs(args, kwargs) -> MultipleInputLayerAttributes:
    if "tensors" in kwargs:
        tensors = kwargs["tensors"]
    else:
        tensors = args[0]
    axis = kwargs.get("dim", 0 if len(args) < 2 else args[1])
    return MultipleInputLayerAttributes(axis=axis, num_inputs=len(tensors))


def _get_kwargs_shifted(args_names, args, kwargs, shift=1):
    res_kwargs = {}
    for idx, arg_name in enumerate(args_names):
        res_kwargs[arg_name] = kwargs[arg_name] if arg_name in kwargs else args[idx + shift]
    return res_kwargs


def _get_const_attrs_from_args_kwargs(args, _) -> ConstantLayerAttributes:
    name = "Unknown"
    shape = []
    if args and isinstance(args[0], TracedParameter):
        name = args[0].name
        shape = args[0].shape
    return ConstantLayerAttributes(name, shape)


def apply_args_defaults(
    args: List[Any], kwargs: Dict[str, Any], args_signature=List[Union[str, Tuple[str, Any]]]
) -> Dict[str, Any]:
    """
    Combines positional arguments (`args`) and keyword arguments (`kwargs`)
    according to the provided `args_signature`.

    The `args_signature` is a list that defines the expected arguments.
    Each element in the list can be either:

    - string: This represents the name of an argument expected to be a positional argument.
    - tuple: This represents the name and default value of an argument.
        - The first element in the tuple is the argument name.
        - The second element in the tuple is the default value.

    :param args: List of positional arguments.
    :param kwargs: Dictionary of keyword arguments.
    :param args_signature: List defining the expected arguments as described above.

    :return: A dictionary combining arguments from `args` and `kwargs` according to the `args_signature`.
    """
    # Manual defines function signature necessary because inspection of torch function is not available
    # https://github.com/pytorch/pytorch/issues/74539

    args_dict: Dict[str, Any] = dict()
    for idx, arg_desc in enumerate(args_signature):
        if isinstance(arg_desc, str):
            if arg_desc in kwargs:
                args_dict[arg_desc] = kwargs[arg_desc]
            elif idx < len(args):
                args_dict[arg_desc] = args[idx]
            else:
                raise ValueError("Incorrect args_signature, can not by applied to function arguments.")
        elif isinstance(arg_desc, Tuple):
            arg_name, default = arg_desc
            args_dict[arg_name] = kwargs.get(arg_name, args[idx] if idx < len(args) else default)
        else:
            raise ValueError("Incorrect args_signature, element of list should be str or tuple.")
    return args_dict


GENERIC_WEIGHT_FUNC_SIGNATURE = ["input", "weight"]
LINEAR_FUNC_SIGNATURE = ["input", "weight", ("bias", None)]
CONV_FUNC_SIGNATURE = ["input", "weight", ("bias", None), ("stride", 1), ("padding", 0), ("dilation", 1), ("groups", 1)]
CONV_TRANSPOSE_FUNC_SIGNATURE = [
    "input",
    "weight",
    ("bias", None),
    ("stride", 1),
    ("padding", 0),
    ("output_padding", 0),
    ("groups", 1),
    ("dilation", 1),
]
BATCH_NORM_FUNC_SIGNATURE = [
    "input",
    "running_mean",
    "running_var",
    ("weight", None),
    ("bias", None),
    ("training", False),
    ("momentum", 0.1),
    ("eps", 1e-5),
]
GROUP_NORM_FUNC_SIGNATURE = [
    "input",
    "num_groups",
    ("weight", None),
    ("bias", None),
    ("eps", None),
]
LAYER_NORM_FUNC_SIGNATURE = [
    "input",
    "normalized_shape",
    ("weight", None),
    ("bias", None),
    ("eps", 1e-05),
    ("cudnn_enable", True),
]


def _get_conv_attrs_from_args_kwargs(args: List[Any], kwargs: Dict[str, Any]) -> ConvolutionLayerAttributes:
    args_dict = apply_args_defaults(args, kwargs, CONV_FUNC_SIGNATURE)

    kernel_size = tuple(args_dict["weight"].shape[2:])
    in_channels = args_dict["weight"].shape[1] * args_dict["groups"]
    out_channels = args_dict["weight"].shape[0]

    return ConvolutionLayerAttributes(
        weight_requires_grad=args_dict["weight"].requires_grad,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=args_dict["stride"],
        dilations=args_dict["dilation"],
        groups=args_dict["groups"],
        transpose=False,
        padding_values=args_dict["padding"],
        with_bias=args_dict["bias"] is not None,
    )


def _get_conv_transpose_attrs_from_args_kwargs(args: List[Any], kwargs: Dict[str, Any]) -> ConvolutionLayerAttributes:
    args_dict = apply_args_defaults(args, kwargs, CONV_TRANSPOSE_FUNC_SIGNATURE)

    kernel_size = tuple(args_dict["weight"].shape[2:])
    in_channels = args_dict["weight"].shape[0]
    out_channels = args_dict["weight"].shape[1] * args_dict["groups"]

    return ConvolutionLayerAttributes(
        weight_requires_grad=args_dict["weight"].requires_grad,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=args_dict["stride"],
        dilations=args_dict["dilation"],
        groups=args_dict["groups"],
        transpose=True,
        padding_values=args_dict["padding"],
        with_bias=args_dict["bias"] is not None,
        output_padding_values=args_dict["output_padding"],
    )


def _get_linear_attrs_from_args_kwargs(args, kwargs) -> LinearLayerAttributes:
    args_dict = apply_args_defaults(args, kwargs, LINEAR_FUNC_SIGNATURE)
    return LinearLayerAttributes(
        weight_requires_grad=args_dict["weight"].requires_grad,
        in_features=args_dict["weight"].shape[1],
        out_features=args_dict["weight"].shape[0],
        with_bias=args_dict["bias"] is not None,
    )


def _get_batchnorm_attrs_from_args_kwargs(args, kwargs):
    args_dict = apply_args_defaults(args, kwargs, BATCH_NORM_FUNC_SIGNATURE)
    return GenericWeightedLayerAttributes(
        weight_requires_grad=False if args_dict["weight"] is None else args_dict["weight"].requires_grad,
        weight_shape=[] if args_dict["weight"] is None else args_dict["weight"].shape,
        filter_dimension_idx=0,
        with_bias=args_dict["bias"] is not None,
    )


def _get_embedding_attrs_from_args_kwargs(args, kwargs):
    args_dict = apply_args_defaults(args, kwargs, GENERIC_WEIGHT_FUNC_SIGNATURE)
    return GenericWeightedLayerAttributes(
        weight_requires_grad=args_dict["weight"].requires_grad,
        weight_shape=args_dict["weight"].shape,
        filter_dimension_idx=0,
        with_bias=False,
    )


def _get_group_norm_attrs_from_args_kwargs(args, kwargs):
    args_dict = apply_args_defaults(args, kwargs, GROUP_NORM_FUNC_SIGNATURE)
    return GroupNormLayerAttributes(
        weight_requires_grad=args_dict["weight"].requires_grad,
        num_channels=args_dict["weight"].shape[0],
        num_groups=args_dict["num_groups"],
    )


def _get_layer_norm_attrs_from_args_kwargs(args, kwargs):
    args_dict = apply_args_defaults(args, kwargs, LAYER_NORM_FUNC_SIGNATURE)
    return GenericWeightedLayerAttributes(
        weight_requires_grad=args_dict["weight"].requires_grad,
        weight_shape=args_dict["weight"].shape,
        filter_dimension_idx=0,
        with_bias=args_dict["bias"] is not None,
    )
