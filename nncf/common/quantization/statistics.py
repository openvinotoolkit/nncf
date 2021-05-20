"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import Dict, Optional

from nncf.api.statistics import Statistics
from nncf.common.utils.helpers import create_table


def _proportion_str(num: int, total_count: int):
    percentage = 100 * (num / max(total_count, 1))
    return f'{percentage:.2f} % ({num} / {total_count})'


class MemoryConsumptionStatistics(Statistics):
    """
    Contains statistics of the memory consumption.
    """

    def __init__(self,
                 fp32_weight_size: int = 0,
                 quantized_weight_size: int = 0,
                 max_fp32_activation_size: int = 0,
                 max_compressed_activation_size: int = 0,
                 weight_memory_consumption_decrease: float = 0.0):
        """
        Initializes statistics of the memory consumption.

        :param fp32_weight_size: Memory consumption for full-precision weights (Mbyte).
        :param quantized_weight_size: Memory consumption for quantized weights (Mbyte).
        :param max_fp32_activation_size: Max memory consumption for an activation
            tensor in FP32 model (Mbyte).
        :param max_compressed_activation_size: Max memory consumption for an activation
            tensor in compressed model (Mbyte).
        :param weight_memory_consumption_decrease: Memory consumption decrease for weights.
        """
        self.fp32_weight_size = fp32_weight_size
        self.quantized_weight_size = quantized_weight_size
        self.max_fp32_activation_size = max_fp32_activation_size
        self.max_compressed_activation_size = max_compressed_activation_size
        self.weight_memory_consumption_decrease = weight_memory_consumption_decrease

    def to_str(self) -> str:
        memory_consumption_string = create_table(
            header=['Statistic\'s name', 'Value'],
            rows=[
                ['Memory consumption for full-precision weights (Mbyte)', self.fp32_weight_size],
                ['Memory consumption for quantized weights (Mbyte)', self.quantized_weight_size],
                [
                    'Max memory consumption for an activation tensor in FP32 model (Mbyte)',
                    self.max_fp32_activation_size
                ],
                [
                    'Max memory consumption for an activation tensor in compressed model (Mbyte)',
                    self.max_compressed_activation_size
                ],
                ['Memory consumption decrease for weights', self.weight_memory_consumption_decrease],
            ]
        )

        pretty_string = f'Statistics of the memory consumption:\n{memory_consumption_string}'
        return pretty_string


class QuantizersCounter:
    def __init__(self,
                 num_symmetric: int = 0,
                 num_asymmetric: int = 0,
                 num_signed: int = 0,
                 num_unsigned: int = 0,
                 num_per_tensor: int = 0,
                 num_per_channel: int = 0):
        """
        Initializes quantizers counter.

        :param num_symmetric: Number of symmetric quantizers.
        :param num_asymmetric: Number of asymmetric quantizers.
        :param num_signed: Number of signed quantizers.
        :param num_unsigned: Number of unsigned quantizers.
        :param num_per_tensor: Number of per-tensor quantizers.
        :param num_per_channel: Number of per-channel quantizers.
        """
        self.num_symmetric = num_symmetric
        self.num_asymmetric = num_asymmetric
        self.num_signed = num_signed
        self.num_unsigned = num_unsigned
        self.num_per_tensor = num_per_tensor
        self.num_per_channel = num_per_channel


class QuantizationShareStatistics(Statistics):
    """
    Contains statistics of the quantization share.
    """

    def __init__(self,
                 wq_total_num: int,
                 aq_total_num: int,
                 wq_potential_num: int,
                 aq_potential_num: int,
                 wq_counter: QuantizersCounter,
                 aq_counter: QuantizersCounter):
        """
        Initializes statistics of the quantization share.

        :param wq_total_num: Total number of weight quantizers.
        :param aq_total_num: Total number of activation quantizers.
        :param wq_potential_num: Potential number of weight quantizers.
        :param aq_potential_num: Potential number of activation quantizers.
        :param wq_counter: Weight quantizers counter.
        :param aq_counter: Activation quantizers counter.
        """
        self.wq_total_num = wq_total_num
        self.aq_total_num = aq_total_num
        self.wq_potential_num = wq_potential_num
        self.aq_potential_num = aq_potential_num
        self.wq_counter = wq_counter
        self.aq_counter = aq_counter

    def to_str(self) -> str:
        mapping = [
            ('num_symmetric', 'Symmetric'),
            ('num_asymmetric', 'Asymmetric'),
            ('num_signed', 'Signed'),
            ('num_unsigned', 'Unsigned'),
            ('num_per_tensor', 'Per-tensor'),
            ('num_per_channel', 'Per-channel'),
        ]

        groups = [
            ('WQ', self.wq_counter, self.wq_total_num),
            ('AQ', self.aq_counter, self.aq_total_num),
        ]

        # Table creation
        header = ['Statistic\'s name', 'Value']
        rows = []
        for q_type, counter, total_num in groups:
            for attr_name, pretty_name in mapping:
                statistic_name = f'{pretty_name} {q_type}s / All placed {q_type}s'
                num = getattr(counter, attr_name)
                rows.append([statistic_name, _proportion_str(num, total_num)])
        rows.append(['Placed WQs / Potential WQs', _proportion_str(self.wq_total_num, self.wq_potential_num)])
        rows.append(['Placed AQs / Potential AQs', _proportion_str(self.aq_total_num, self.aq_potential_num)])

        qshare_string = create_table(header, rows)
        pretty_string = f'Statistics of the quantization share:\n{qshare_string}'
        return pretty_string


class BitwidthDistributionStatistics(Statistics):
    """
    Contains statistics of the bitwidth distribution.
    """

    def __init__(self,
                 num_wq_per_bitwidth: Dict[int, int],
                 num_aq_per_bitwidth: Dict[int, int]):
        """
        Initializes bitwidth distribution statistics.

        :param num_wq_per_bitwidth: Number of weight quantizers per bit width.
        :param num_aq_per_bitwidth: Number of activation quantizers per bit width.
        """
        self.num_wq_per_bitwidth = num_wq_per_bitwidth
        self.num_aq_per_bitwidth = num_aq_per_bitwidth

    def to_str(self) -> str:
        wq_total_num = sum(self.num_wq_per_bitwidth.values())
        aq_total_num = sum(self.num_aq_per_bitwidth.values())
        q_total_num = wq_total_num + aq_total_num

        bitwidths = self.num_wq_per_bitwidth.keys() | self.num_aq_per_bitwidth.keys()  # union of all bitwidths
        bitwidths = sorted(bitwidths, reverse=True)

        # Table creation
        header = ['Num bits (N)', 'N-bits WQs / Placed WQs', 'N-bits AQs / Placed AQs', 'N-bits Qs / Placed Qs']
        rows = []
        for bitwidth in bitwidths:
            wq_num = self.num_wq_per_bitwidth.get(bitwidth, 0)  # for current bitwidth
            aq_num = self.num_aq_per_bitwidth.get(bitwidth, 0)  # for current bitwidth
            q_num = wq_num + aq_num  # for current bitwidth

            rows.append([
                bitwidth,
                _proportion_str(wq_num, wq_total_num),
                _proportion_str(aq_num, aq_total_num),
                _proportion_str(q_num, q_total_num)
            ])

        distribution_string = create_table(header, rows)
        pretty_string = f'Statistics of the bitwidth distribution:\n{distribution_string}'
        return pretty_string


class QuantizationConfigurationStatistics(Statistics):
    """
    Contains statistics of the quantization configuration.
    """

    def __init__(self, quantized_edges_in_cfg: int, total_edges_in_cfg: int):
        """
        Initializes statistics of the quantization configuration.

        :param quantized_edges_in_cfg: Number of quantized edges in quantization configuration.
        :param total_edges_in_cfg: Total number of edges in quantization configuration.
        """
        self.quantized_edges_in_cfg = quantized_edges_in_cfg
        self.total_edges_in_cfg = total_edges_in_cfg

    def to_str(self) -> str:
        header = ['Statistic\'s name', 'Value']
        rows = [
            [
                'Share edges of the quantized data path',
                _proportion_str(self.quantized_edges_in_cfg, self.total_edges_in_cfg)
            ]
        ]
        qc_string = create_table(header, rows)
        pretty_string = f'Statistics of the quantization configuration:\n{qc_string}'
        return pretty_string


class QuantizationStatistics(Statistics):
    """
    Contains statistics of the quantization algorithm.
    """

    def __init__(self,
                 ratio_of_enabled_quantizations: float,
                 quantization_share_statistics: Optional[QuantizationShareStatistics] = None,
                 bitwidth_distribution_statistics: Optional[BitwidthDistributionStatistics] = None,
                 memory_consumption_statistics: Optional[MemoryConsumptionStatistics] = None,
                 quantization_configuration_statistics: Optional[QuantizationConfigurationStatistics] = None):
        """
        Initializes statistics of the quantization algorithm.

        :param ratio_of_enabled_quantizations: Ratio of enabled quantizations.
        :param quantization_share_statistics: Statistics of the quantization share.
        :param bitwidth_distribution_statistics: Statistics of the bitwidth distribution.
        :param memory_consumption_statistics: Statistics of the memory consumption.
        :param quantization_configuration_statistics: Statistics of the quantization configuration.
        """
        self.ratio_of_enabled_quantizations = ratio_of_enabled_quantizations
        self.quantization_share_statistics = quantization_share_statistics
        self.bitwidth_distribution_statistics = bitwidth_distribution_statistics
        self.memory_consumption_statistics = memory_consumption_statistics
        self.quantization_configuration_statistics = quantization_configuration_statistics

    def to_str(self) -> str:
        pretty_strings = []

        table_string = create_table(
            header=['Statistic\'s name', 'Value'],
            rows=[['Ratio of enabled quantizations', self.ratio_of_enabled_quantizations]]
        )
        pretty_strings.append(f'Statistics of the quantization algorithm:\n{table_string}')

        statistics = [
            self.quantization_share_statistics,
            self.bitwidth_distribution_statistics,
            self.memory_consumption_statistics,
            self.quantization_configuration_statistics
        ]

        for stats in statistics:
            if stats:
                pretty_strings.append(stats.to_str())

        pretty_string = '\n\n'.join(pretty_strings)
        return pretty_string
