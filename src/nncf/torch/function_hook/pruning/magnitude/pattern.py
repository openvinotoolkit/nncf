# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Structured M:N sparsity patterns for magnitude-based pruning.

In an M:N sparsity pattern every consecutive group of N weights contains exactly M zeros,
where the zeroed weights are selected by magnitude. The 2:4 pattern (two zeros in every
group of four weights) is the most widely supported one by runtimes and hardware.
"""

from dataclasses import dataclass

import torch

import nncf
from nncf.parameters import PruneMode
from nncf.torch.graph.operator_metatypes import PTConv1dMetatype
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTConv3dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose1dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose2dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose3dMetatype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv1dSubtype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv2dSubtype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv3dSubtype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.graph.operator_metatypes import PTMatMulMetatype
from nncf.torch.graph.operator_metatypes import PTOperatorMetatype
from nncf.torch.model_graph_manager import get_weight_compression_reduction_axes

# Metatypes whose weight tensors can currently be pruned with a structured M:N pattern.
# Depthwise convolutions are excluded because their output-channel dimension has size 1,
# so no group of more than one weight can be formed along it.
STRUCTURED_PRUNING_SUPPORTED_METATYPES = [
    PTConv1dMetatype,
    PTConv2dMetatype,
    PTConv3dMetatype,
    PTConvTranspose1dMetatype,
    PTConvTranspose2dMetatype,
    PTConvTranspose3dMetatype,
    PTDepthwiseConv1dSubtype,
    PTDepthwiseConv2dSubtype,
    PTDepthwiseConv3dSubtype,
    PTLinearMetatype,
    PTMatMulMetatype,
]


@dataclass(frozen=True)
class SparsityPattern:
    """
    Structured M:N sparsity pattern.

    For every consecutive group of ``n`` weights, ``m`` weights with the smallest absolute
    values are zeroed out and the remaining ``n - m`` weights are kept. The invariant holds
    independently for every group of the weight tensor.

    :param m: Number of weights to prune in each group of ``n`` weights.
    :param n: Size of the group of consecutive weights.
    """

    m: int
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0 or self.m <= 0:
            msg = f"Sparsity pattern values should be positive, got m={self.m}, n={self.n}."
            raise nncf.ValidationError(msg)
        if self.m >= self.n:
            msg = (
                "In the M:N sparsity pattern, m should be less than n so that at least one weight "
                f"is retained in every group, got m={self.m}, n={self.n}."
            )
            raise nncf.ValidationError(msg)

    def __str__(self) -> str:
        return f"{self.m}:{self.n}"


def get_sparsity_pattern(mode: PruneMode) -> SparsityPattern:
    """
    Returns the sparsity pattern corresponding to the given structured pruning mode.

    :param mode: The pruning mode.
    :return: The sparsity pattern.
    """
    if mode == PruneMode.STRUCTURED_MAGNITUDE_2_4:
        return SparsityPattern(m=2, n=4)
    msg = f"Pruning mode {mode} does not define a structured sparsity pattern."
    raise nncf.InternalError(msg)


def get_structured_pattern_groups(
    data: torch.Tensor,
    metatype: type[PTOperatorMetatype],
    weight_port_id: int,
    pattern: SparsityPattern,
) -> torch.Tensor:
    """
    Reshapes the weight tensor into a 3D view `(out_channels, groups, pattern.n)` so that the last dimension
    enumerates all groups of consecutive weights of size `pattern.n` along the structural dimension of
    the tensor, i.e. the dimension along which the output channels are laid out contiguously.

    The structural dimension is derived from the same channel and reduction axes that the weight
    compression algorithms use, so the layout is consistent across NNCF pruning and compression:
    for Linear and MatMul weights the structural dimension is the output-channel (row) dimension,
    for convolution weights it is the output-channel-by-spatial-position layout. In both cases the
    last dimension of the reshaped view enumerates the consecutive weights inside one output channel.

    :param data: The weight tensor.
    :param metatype: The metatype of the operation that consumes the weight.
    :param weight_port_id: The input port id of the weight tensor.
    :param pattern: The structured sparsity pattern.
    :return: The reshaped view of the weight tensor with shape
        `(num_out_channels, num_weights_per_out_channel / pattern.n, pattern.n)`.
    :raises nncf.ValidationError: If the structural dimension of the tensor is not divisible by `pattern.n`.
    """
    ndims = data.dim()
    reduction_axes = get_weight_compression_reduction_axes(metatype, weight_port_id, ndims)
    channel_axes = [axis for axis in range(ndims) if axis not in reduction_axes]

    if len(channel_axes) != 1:
        msg = (
            f"Cannot determine the structural dimension for a weight of metatype {metatype} "
            f"with shape {tuple(data.shape)}: expected exactly one channel axis, got {channel_axes}."
        )
        raise nncf.ValidationError(msg)

    channel_axis = channel_axes[0]
    # Move the channel axis to the front and flatten the remaining (reduction) axes into one
    # trailing dimension, so that consecutive weights inside one output channel are contiguous.
    # The permutation is required because `reshape` flattens the remaining dimensions in row-major
    # order, which keeps the weights of every output channel contiguous in the trailing dimension.
    permutation = [channel_axis] + [axis for axis in range(ndims) if axis != channel_axis]
    view = data.permute(*permutation).reshape(data.shape[channel_axis], -1)

    num_out_channels, num_weights_per_out_channel = view.shape
    if num_weights_per_out_channel % pattern.n != 0:
        msg = (
            f"The structural dimension of the weight tensor with shape {tuple(data.shape)} "
            f"is not divisible by the group size {pattern.n} of the {pattern} sparsity pattern: "
            f"{num_weights_per_out_channel} weights per output channel cannot be split "
            f"into groups of {pattern.n} weights. "
            "Use nncf.IgnoredScope to exclude this operation from pruning."
        )
        raise nncf.ValidationError(msg)

    return view.reshape(num_out_channels, num_weights_per_out_channel // pattern.n, pattern.n)
