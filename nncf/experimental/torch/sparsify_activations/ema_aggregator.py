# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import nncf.tensor.functions as fns
from nncf.experimental.common.tensor_statistics.collectors import AggregationAxes
from nncf.experimental.common.tensor_statistics.collectors import OnlineAggregatorBase
from nncf.tensor import Tensor


# TODO: add tests
class EMAAggregator(OnlineAggregatorBase):
    def __init__(
        self,
        alpha: float,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ):
        self._alpha = alpha
        super().__init__(aggregation_axes=(0,), num_samples=num_samples, window_size=window_size)

    def _aggregation_fn(self, stacked_value: Tensor, axis: AggregationAxes, keepdims: bool) -> Tensor:
        if self._collected_samples == 0:
            return stacked_value
        else:
            beta = 1.0 - self._alpha
            new_value = fns.expand_dims(stacked_value[0], 0)
            old_value = fns.expand_dims(stacked_value[1], 0)
            return new_value * self._alpha + old_value * beta * (1 - beta**self._collected_samples) / (
                1 - beta ** (self._collected_samples + 1)
            )
