"""
 Copyright (c) 2022 Intel Corporation
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

from numbers import Number
from typing import Dict
from nncf.api.statistics import Statistics

from nncf.common.utils.tensorboard import convert_to_dict


class FracBitsStatistics(Statistics):
    def __init__(self, states: Dict[str, Number]) -> None:
        super().__init__()
        self.data = states

    def to_str(self) -> str:
        return str(self.data)


@convert_to_dict.register(FracBitsStatistics)
def _convert_to_dict(stats: FracBitsStatistics, algorithm_name: str):
    tensorboard_stats = {
        algorithm_name + "/" + k: v for k, v in stats.data.items()
    }
    return tensorboard_stats
