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
from dataclasses import field
from itertools import product
from typing import Any

from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.quantization.structs import QuantizerConfig
from nncf.parameters import StrEnum

SCALES = "scales"
UNIFIED = "unified"
ADJUST_PADDING = "adjust_padding"


class Granularity(StrEnum):
    PER_CHANNEL = "per_channel"
    PER_TENSOR = "per_tensor"


@dataclass(frozen=True, kw_only=True, slots=True)
class QConfigSpace:
    """
    A class to represent the configuration space for quantization.

    :param bits: Number of bits for quantization.
    :param mode: Available quantization schemes.
    :param granularity: Granularity options for quantization.
    :param narrow_range: Indicates narrow range quantization.
    :param signedness_to_force: Optional signedness enforcement.
    """

    bits: int
    mode: tuple[QuantizationScheme, ...]
    granularity: tuple[Granularity, ...]
    narrow_range: tuple[bool, ...]
    signedness_to_force: bool | None = None

    def get_all_qconfigs(self) -> list[QuantizerConfig]:
        """
        Generate a list of all possible QuantizerConfig instances based on the current
        settings of mode, granularity, narrow_range, and other parameters.

        :return: A list of QuantizerConfig objects, each representing
            a unique combination of the quantization parameters.
        """
        ret = []
        for mode, granularity, narrow_range in product(self.mode, self.granularity, self.narrow_range):
            ret.append(
                QuantizerConfig(
                    num_bits=self.bits,
                    mode=mode,
                    per_channel=granularity == Granularity.PER_CHANNEL,
                    narrow_range=narrow_range,
                    signedness_to_force=self.signedness_to_force,
                )
            )
        return ret


@dataclass(frozen=True, kw_only=True, slots=True)
class OpDesc:
    """
    Represents the description of quantization schemes applicable for activations and weights of operation.

    :param type: The type of the operation.
    :param activations: A tuple containing the quantization configuration for the activations of the operation.
    :param weights: A tuple containing the quantization configuration for the weights of the operation.
    :param attributes: A dictionary of additional attributes.
    """

    type: str
    activations: tuple[QConfigSpace, ...] = field(default_factory=tuple)
    weights: tuple[QConfigSpace, ...] = field(default_factory=tuple)
    attributes: dict[str, Any] = field(default_factory=dict)
