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

from beta.nncf.tensorflow.callbacks.statistics_callback import StatisticsCallback
from beta.nncf.tensorflow.sparsity.utils import convert_raw_to_printable
from beta.nncf.tensorflow.sparsity.utils import prepare_for_tensorboard


class PruningStatisticsCallback(StatisticsCallback):
    """
    Callback for logging cruning compression statistics to tensorboard and stdout
    """

    def _prepare_for_tensorboard(self, raw_statistics: dict) -> dict:
        prefix = 'pruning'
        rate_abbreviation = 'PR'
        return prepare_for_tensorboard(raw_statistics, prefix, rate_abbreviation)

    def _convert_raw_to_printable(self, raw_statistics: dict) -> dict:
        prefix = 'pruning'
        header = ['Name', 'Weight\'s Shape', 'Mask Shape', 'PR']
        return convert_raw_to_printable(raw_statistics, prefix, header)
