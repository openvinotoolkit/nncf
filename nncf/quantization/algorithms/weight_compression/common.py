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

from dataclasses import dataclass
from typing import Any, Optional, Union

from nncf.errors import InvalidGroupSizeError
from nncf.errors import UnsupportedModelError
from nncf.tensor import Tensor

ReductionAxes = Union[int, tuple[int, ...]]


@dataclass
class Codebook:
    """
    Codebook parameters for weight compression.
    :param codebook: The initial codebook for compression.
    :param dst_type: The destination type for the codebook.
    """

    codebook: Optional[Tensor] = None
    dst_type: Optional[Any] = None


@dataclass
class CompressedWeight:
    """
    Compressed weight and decompression parameters.

    :param tensor: The tensor with compressed weight.
    :param scale: The decompression scale, in practice it is dequantization scale for the quantization.
    :param zero_point: The zero-point, it is the value of the compression type corresponding to the value 0
        in the non-compression realm. Applicable for INT quantization.
    :param codebook: The codebook (LUT) for the weight compression. Applicable for vector quantization
    """

    tensor: Optional[Tensor] = None
    scale: Optional[Tensor] = None
    zero_point: Optional[Tensor] = None
    codebook: Optional[Codebook] = None

    def is_codebook(self):
        """
        Check if the compressed weight is a codebook.

        :return: True if the compressed weight is a codebook, False otherwise.
        """
        return self.codebook is not None and self.tensor is not None and self.scale is not None


def reshape_weight_for_grouped_quantization(
    weight: Tensor, reduction_axes: ReductionAxes, group_size: int
) -> tuple[Tensor, int]:
    """
    Reshapes weight for group-wise quantization and return a reduction axis for collecting statistics per group
    dimension. Having a transposed weight with shapes [c_out, c_in] and group size = 128, shape of reshaped weight is
    [c_out, c_in // 128, 128], reduction axis = 1 and the returned reduction axis = 2.

    :param weight: Weight array to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
    :return: reshaped weight and new reduction axis.
    """
    assert group_size != -1
    if isinstance(reduction_axes, tuple) and len(reduction_axes) == 1:
        reduction_axes = reduction_axes[0]
    if not isinstance(reduction_axes, int):
        msg = f"Group-wise quantization expects a single reduction axis, but given: {reduction_axes}."
        raise UnsupportedModelError(msg)
    channel_size = weight.shape[reduction_axes]
    if channel_size % group_size != 0:
        msg = f"Channel size {channel_size} should be divisible by size of group {group_size}."
        raise InvalidGroupSizeError(msg)

    num_groups_per_channel = channel_size // group_size
    shape = list(weight.shape)  # [a1, r, a2] - "r" refers to number of channels along reduction axis
    shape[reduction_axes : reduction_axes + 1] = (num_groups_per_channel, group_size)
    reshaped_weight = weight.reshape(shape)
    reduction_axes += 1
    return reshaped_weight, reduction_axes
