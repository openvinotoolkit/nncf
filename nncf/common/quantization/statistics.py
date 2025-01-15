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

from typing import Dict, List, Optional

from nncf.api.statistics import Statistics
from nncf.common.utils.api_marker import api
from nncf.common.utils.helpers import create_table


def _proportion_str(num: int, total_count: int) -> str:
    percentage = 100 * (num / max(total_count, 1))
    return f"{percentage:.2f} % ({num} / {total_count})"


class QuantizersCounter:
    def __init__(
        self,
        num_symmetric: int = 0,
        num_asymmetric: int = 0,
        num_signed: int = 0,
        num_unsigned: int = 0,
        num_per_tensor: int = 0,
        num_per_channel: int = 0,
        total_count: int = 0,
        potential_count: Optional[int] = None,
    ):
        """
        Initializes quantizers counter.

        :param num_symmetric: Number of symmetric quantizers.
        :param num_asymmetric: Number of asymmetric quantizers.
        :param num_signed: Number of signed quantizers.
        :param num_unsigned: Number of unsigned quantizers.
        :param num_per_tensor: Number of per-tensor quantizers.
        :param num_per_channel: Number of per-channel quantizers.
        :param total_count: Total count of quantizers.
        :param potential_count: Count of potential quantizers.
        """
        self.num_symmetric = num_symmetric
        self.num_asymmetric = num_asymmetric
        self.num_signed = num_signed
        self.num_unsigned = num_unsigned
        self.num_per_tensor = num_per_tensor
        self.num_per_channel = num_per_channel
        self.total_count = total_count
        self.potential_count = potential_count


def _quantizers_counter_to_rows(counter: QuantizersCounter, qt: str) -> List[List[str]]:
    """
    Converts the counter of quantizers to rows.

    :param counter: Counter of quantizers.
    :param qt: Type of the counter. Takes one of the following values:
        - `WQ` - for the counter of the quantizers for weight.
        - `AQ` - for the counter of the quantizers for activation.
    :return: List of rows.
    """
    rows = [
        [
            f"Symmetric {qt}s / All placed {qt}s",
            _proportion_str(counter.num_symmetric, counter.total_count),
        ],
        [
            f"Asymmetric {qt}s / All placed {qt}s",
            _proportion_str(counter.num_asymmetric, counter.total_count),
        ],
        [
            f"Signed {qt}s / All placed {qt}s",
            _proportion_str(counter.num_signed, counter.total_count),
        ],
        [
            f"Unsigned {qt}s / All placed {qt}s",
            _proportion_str(counter.num_unsigned, counter.total_count),
        ],
        [
            f"Per-tensor {qt}s / All placed {qt}s",
            _proportion_str(counter.num_per_tensor, counter.total_count),
        ],
        [
            f"Per-channel {qt}s / All placed {qt}s",
            _proportion_str(counter.num_per_channel, counter.total_count),
        ],
    ]

    if counter.potential_count:
        rows.append([f"Placed {qt}s / Potential {qt}s", _proportion_str(counter.total_count, counter.potential_count)])

    return rows


@api()
class QuantizationStatistics(Statistics):
    """
    Contains statistics of the quantization algorithm. These statistics include:

    * Information about the share of the quantization, such as:

      * Percentage of symmetric/asymmetric/per-channel/per-tensor weight quantizers relative to the number of placed
        weight quantizers.
      * Percentage of symmetric/asymmetric/per-channel/per-tensor non-weight quantizers relative to the number of
        placed non weight quantizers.
      * Percentage of weight quantizers and non-weight quantizers for each precision relative to the number
        of potential quantizers/placed quantizers.

    * Information about the distribution of the bitwidth of the quantizers.
    * Ratio of enabled quantization.

    .. note:: The maximum possible number of potential quantizers depends on the presence of ignored scopes and the
      mode of quantizer setup that is used at the time of collecting the metric.

    :param wq_counter: Weight quantizers counter.
    :param aq_counter: Activation quantizers counter.
    :param num_wq_per_bitwidth: Number of weight quantizers per bit width.
    :param num_aq_per_bitwidth: Number of activation quantizers per bit width.
    :param ratio_of_enabled_quantizations: Ratio of enabled quantizations.
    """

    def __init__(
        self,
        wq_counter: QuantizersCounter,
        aq_counter: QuantizersCounter,
        num_wq_per_bitwidth: Dict[int, int],
        num_aq_per_bitwidth: Dict[int, int],
        ratio_of_enabled_quantizations: float,
    ):
        self.wq_counter = wq_counter
        self.aq_counter = aq_counter
        self.num_wq_per_bitwidth = num_wq_per_bitwidth
        self.num_aq_per_bitwidth = num_aq_per_bitwidth
        self.ratio_of_enabled_quantizations = ratio_of_enabled_quantizations

    def to_str(self) -> str:
        pretty_strings = []

        table = create_table(
            header=["Statistic's name", "Value"],
            rows=[["Ratio of enabled quantizations", self.ratio_of_enabled_quantizations]],
        )

        pretty_strings.append(f"Statistics of the quantization algorithm:\n{table}")
        pretty_strings.append(self._get_quantization_share_str())
        pretty_strings.append(self._get_bitwidth_distribution_str())
        pretty_string = "\n\n".join(pretty_strings)
        return pretty_string

    def _get_quantization_share_str(self) -> str:
        header = ["Statistic's name", "Value"]

        rows = []
        rows.extend(_quantizers_counter_to_rows(self.wq_counter, "WQ"))
        rows.extend(_quantizers_counter_to_rows(self.aq_counter, "AQ"))

        table = create_table(header, rows)
        pretty_string = f"Statistics of the quantization share:\n{table}"
        return pretty_string

    def _get_bitwidth_distribution_str(self) -> str:
        wq_total_num = sum(self.num_wq_per_bitwidth.values())
        aq_total_num = sum(self.num_aq_per_bitwidth.values())
        q_total_num = wq_total_num + aq_total_num

        bitwidths = self.num_wq_per_bitwidth.keys() | self.num_aq_per_bitwidth.keys()  # union of all bitwidths
        bitwidths_sorted = sorted(bitwidths, reverse=True)

        # Table creation
        header = ["Num bits (N)", "N-bits WQs / Placed WQs", "N-bits AQs / Placed AQs", "N-bits Qs / Placed Qs"]
        rows = []
        for bitwidth in bitwidths_sorted:
            wq_num = self.num_wq_per_bitwidth.get(bitwidth, 0)  # for current bitwidth
            aq_num = self.num_aq_per_bitwidth.get(bitwidth, 0)  # for current bitwidth
            q_num = wq_num + aq_num  # for current bitwidth

            rows.append(
                [
                    bitwidth,
                    _proportion_str(wq_num, wq_total_num),
                    _proportion_str(aq_num, aq_total_num),
                    _proportion_str(q_num, q_total_num),
                ]
            )

        table = create_table(header, rows)
        pretty_string = f"Statistics of the bitwidth distribution:\n{table}"
        return pretty_string
