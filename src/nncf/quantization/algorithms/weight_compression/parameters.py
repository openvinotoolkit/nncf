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

from dataclasses import dataclass

from nncf.tensor import Tensor


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

    tensor: Tensor | None = None
    scale: Tensor | None = None
    zero_point: Tensor | None = None
    codebook: Tensor | None = None

    def is_codebook(self):
        """
        Check if the compressed weight is a codebook.

        :return: True if the compressed weight is a codebook, False otherwise.
        """
        return self.codebook is not None and self.tensor is not None and self.scale is not None
