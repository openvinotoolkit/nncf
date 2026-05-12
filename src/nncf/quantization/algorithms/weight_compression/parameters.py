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
from nncf.tensor import functions as fns


@dataclass
class CompressedWeight:
    """
    Compressed weight and decompression parameters.

    :param tensor: The tensor with compressed weight.
    :param scale: The decompression scale, in practice it is dequantization scale for the quantization.
    :param zero_point: The zero-point, it is the value of the compression type corresponding to the value 0
        in the non-compression realm. Applicable for INT quantization.
    :param codebook: The codebook (LUT) for the weight compression. Applicable for vector quantization
    :param global_scale: The tensor-wide (global) scaling factor used in two-level scaling schemes such as NVFP4.
    """

    tensor: Tensor | None = None
    scale: Tensor | None = None
    zero_point: Tensor | None = None
    codebook: Tensor | None = None
    global_scale: Tensor | None = None

    def get_unscaled_tensor(self) -> Tensor | None:
        """
        Returns the quantized weight values before scaling. For non-codebook compression, equals the tensor as-is.
        For codebook compression, returns the codebook values indexed by the tensor.

        :return: Tensor with quantized weight values.
        """
        if self.tensor is not None and self.codebook is not None:
            # Convert fp datatypes like f8e4m3/fp4e2m1 to a numpy fp32 array
            # Then converting the array back to the backend tensor
            codebook = fns.from_numpy(self.codebook.as_numpy_tensor().data, backend=self.tensor.backend)
            return codebook[self.tensor]
        return self.tensor
