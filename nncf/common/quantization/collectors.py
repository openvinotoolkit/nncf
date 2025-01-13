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

from abc import abstractmethod
from collections import Counter
from typing import List, Tuple

from nncf.common.collector import StatisticsCollector
from nncf.common.quantization.statistics import QuantizationStatistics
from nncf.common.quantization.statistics import QuantizersCounter


class QuantizerDescription:
    """
    Contains information about the quantizer.
    """

    def __init__(
        self,
        num_bits: int,
        is_per_channel: bool,
        is_signed: bool,
        is_symmetric: bool,
        is_weight_quantizer: bool,
        is_enabled: bool,
    ):
        """
        Initializes the description of the quantizer.

        :param num_bits: Bitwidth of the quantization.
        :param is_per_channel: `True` for per-channel quantization, `False` for per-tensor.
        :param is_signed: `True` for signed quantization, `False` for unsigned.
        :param is_symmetric: `True` for symmetric quantizer, `False` for asymmetric.
        :param is_weight_quantizer: `True` for weight quantizer, `False` for non-weight.
        :param is_enabled: `True` for enabled quantizer, `False` for disabled.
        """
        self._num_bits = num_bits
        self._is_per_channel = is_per_channel
        self._is_signed = is_signed
        self._is_symmetric = is_symmetric
        self._is_weight_quantizer = is_weight_quantizer
        self._is_enabled = is_enabled

    @property
    def num_bits(self) -> int:
        return self._num_bits

    @property
    def is_per_channel(self) -> bool:
        return self._is_per_channel

    @property
    def is_signed(self) -> bool:
        return self._is_signed

    @property
    def is_symmetric(self) -> bool:
        return self._is_symmetric

    @property
    def is_weight_quantizer(self) -> bool:
        return self._is_weight_quantizer

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled


class QuantizationStatisticsCollector(StatisticsCollector):
    """
    Base class for the quantization statistics collector.
    """

    @abstractmethod
    def _collect_quantizers_descriptions(self) -> List[QuantizerDescription]:
        """
        Collects descriptions of the quantizers.

        :return: Descriptions of the quantizers.
        """

    @abstractmethod
    def _get_potential_quantizers_num(self) -> Tuple[int, int]:
        """
        Returns a potential number of quantizers for weights and activations.

        :return: A tuple (wq_potential_num, aq_potential_num) where
            - `wq_potential_num` is a potential number of quantizers for weights.
            - `aq_potential_num` is a potential number of quantizers for activations.
        """

    def collect(self) -> QuantizationStatistics:
        """
        Collects statistics of the quantization algorithm.

        :return: A statistics of the quantization algorithm.
        """
        quantizers_descriptions = self._collect_quantizers_descriptions()

        wq_counter = QuantizersCounter()
        aq_counter = QuantizersCounter()
        wq_bitwidths = []
        aq_bitwidths = []
        num_enabled_quantizers = 0

        wq_counter.potential_count, aq_counter.potential_count = self._get_potential_quantizers_num()

        for q in quantizers_descriptions:
            if q.is_weight_quantizer:
                counter = wq_counter
                wq_bitwidths.append(q.num_bits)
            else:
                counter = aq_counter
                aq_bitwidths.append(q.num_bits)

            if q.is_per_channel:
                counter.num_per_channel += 1
            else:
                counter.num_per_tensor += 1

            if q.is_signed:
                counter.num_signed += 1
            else:
                counter.num_unsigned += 1

            if q.is_symmetric:
                counter.num_symmetric += 1
            else:
                counter.num_asymmetric += 1

            if q.is_enabled:
                num_enabled_quantizers += 1

            counter.total_count += 1

        num_wq_per_bitwidth = dict(Counter(wq_bitwidths))
        num_aq_per_bitwidth = dict(Counter(aq_bitwidths))

        total_count = wq_counter.total_count + aq_counter.total_count
        ratio_of_enabled_quantizations = 100 * (num_enabled_quantizers / max(total_count, 1))

        return QuantizationStatistics(
            wq_counter, aq_counter, num_wq_per_bitwidth, num_aq_per_bitwidth, ratio_of_enabled_quantizations
        )
