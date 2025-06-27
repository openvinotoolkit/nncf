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
from typing import Any, Optional

from nncf.tensor import Tensor


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
