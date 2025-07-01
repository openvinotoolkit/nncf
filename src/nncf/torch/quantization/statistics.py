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

from nncf.api.statistics import Statistics
from nncf.common.utils.helpers import create_table


def _proportion_str(num: int, total_count: int):
    percentage = 100 * (num / max(total_count, 1))
    return f"{percentage:.2f} % ({num} / {total_count})"


class MemoryConsumptionStatistics(Statistics):
    """
    Contains statistics of the memory consumption.
    """

    def __init__(
        self,
        fp32_weight_size: int = 0,
        quantized_weight_size: int = 0,
        max_fp32_activation_size: int = 0,
        max_compressed_activation_size: int = 0,
        weight_memory_consumption_decrease: float = 0.0,
    ):
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
            header=["Statistic's name", "Value"],
            rows=[
                ["Memory consumption for full-precision weights (Mbyte)", self.fp32_weight_size],
                ["Memory consumption for quantized weights (Mbyte)", self.quantized_weight_size],
                [
                    "Max memory consumption for an activation tensor in FP32 model (Mbyte)",
                    self.max_fp32_activation_size,
                ],
                [
                    "Max memory consumption for an activation tensor in compressed model (Mbyte)",
                    self.max_compressed_activation_size,
                ],
                ["Memory consumption decrease for weights", self.weight_memory_consumption_decrease],
            ],
        )

        pretty_string = f"Statistics of the memory consumption:\n{memory_consumption_string}"
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
        header = ["Statistic's name", "Value"]
        rows = [
            [
                "Share edges of the quantized data path",
                _proportion_str(self.quantized_edges_in_cfg, self.total_edges_in_cfg),
            ]
        ]
        qc_string = create_table(header, rows)
        pretty_string = f"Statistics of the quantization configuration:\n{qc_string}"
        return pretty_string
