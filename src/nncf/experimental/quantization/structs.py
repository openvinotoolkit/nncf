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

from typing import Any, Literal, Optional

from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.quantization.structs import QuantizerConfig
from nncf.config.schemata.defaults import QUANTIZATION_BITS
from nncf.config.schemata.defaults import QUANTIZATION_NARROW_RANGE
from nncf.config.schemata.defaults import QUANTIZATION_PER_CHANNEL
from nncf.tensor.definitions import TensorDataType

IntDtype = Literal[TensorDataType.int8, TensorDataType.uint8]


class ExtendedQuantizerConfig(QuantizerConfig):
    """
    Extended configuration class for quantizers, including destination integer dtype.
    """

    def __init__(
        self,
        num_bits: int = QUANTIZATION_BITS,
        mode: QuantizationScheme = QuantizationScheme.SYMMETRIC,
        signedness_to_force: Optional[bool] = None,
        per_channel: bool = QUANTIZATION_PER_CHANNEL,
        narrow_range: bool = QUANTIZATION_NARROW_RANGE,
        dest_dtype: IntDtype = TensorDataType.int8,
    ):
        """
        :param num_bits: Bitwidth of the quantization.
        :param mode: The mode of quantization (symmetric or asymmetric).
        :param signedness_to_force: True if the quantizer *must* be signed, False if *must* be unsigned,
            None if the signed/unsigned attribute should be determined based on the incoming activation
            statistics during range initialization.
        :param per_channel: True for per-channel quantization, False for per-tensor.
        :param narrow_range: True if the range of quantized values should be narrowed as compared to the
            naive case, False if all 2^`num_bits` quantizations should be used.
        :param dest_dtype: Target integer data type for quantized values.
        """
        super().__init__(num_bits, mode, signedness_to_force, per_channel, narrow_range)
        self.dest_dtype = dest_dtype

    def __str__(self) -> str:
        retval = super().__str__()
        return retval + " DestDtype: {self._dest_dtype}"

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["dest_dtype"] = self.dest_dtype
        return state
