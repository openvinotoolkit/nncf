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
from typing import Callable, Union

import torch

from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.tensor import Tensor
from nncf.torch.function_hook.hook_executor_mode import disable_function_hook_mode
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.return_types import maybe_get_values_from_torch_return_type


class TensorStatisticObservationPoint:
    def __init__(self, target_point: PTTargetPoint, reduction_shapes: set[ReductionAxes] = None):
        self.target_point = target_point
        self.reduction_shapes = reduction_shapes

    def __hash__(self):
        return hash(self.target_point)

    def __eq__(self, other: "TensorStatisticObservationPoint"):
        return self.target_point == other.target_point


def create_register_input_hook(collector: TensorCollector) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Function to create register inputs hook function.

    :param collector: Collector to use in resulting hook.
    :return: Register inputs hook function.
    """

    def register_inputs_hook(x: Union[torch.Tensor, tuple]) -> torch.Tensor:
        """
        Register inputs hook function.

        :parameter x: tensor to register in hook.
        :return: tensor to register in hook.
        """
        with disable_function_hook_mode():
            x_unwrapped = maybe_get_values_from_torch_return_type(x)
            collector.register_input_for_all_reducers(Tensor(x_unwrapped))
        return x

    return register_inputs_hook
