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

from nncf.api.statistics import Statistics
from nncf.common.utils.api_marker import api


@api()
@dataclass
class TrainingLoopStatistics(Statistics):
    """
    Contains statistics related to Accuracy Aware Training Loop
    """

    uncompressed_accuracy: float
    compression_rate: float
    compressed_accuracy: float
    absolute_accuracy_degradation: float
    relative_accuracy_degradation: float
    accuracy_budget: float

    def to_str(self) -> str:
        stats_str = (
            f"Uncompressed model accuracy: {self.uncompressed_accuracy:.4f}\n"
            f"Compressed model accuracy: {self.compressed_accuracy:.4f}\n"
            f"Model compression rate: {self.compression_rate:.4f}\n"
            f"Absolute accuracy drop: {self.absolute_accuracy_degradation:.4f}\n"
            f"Relative accuracy drop: {self.relative_accuracy_degradation:.2f}%\n"
            f"Accuracy budget: {self.accuracy_budget:.4f}"
        )
        return stats_str
