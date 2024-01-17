# Copyright (c) 2023 Intel Corporation
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

from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.return_types import maybe_get_values_from_torch_return_type
from nncf.torch.tensor import PTNNCFTensor


def create_register_input_hook(collector: TensorCollector) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Function to create regiter inputs hook function.

    :param collector: Collector to use in resulting hook.
    :return: Register inputs hook function.
    """

    def register_inputs_hook(x: Union[torch.Tensor, tuple]) -> torch.Tensor:
        """
        Register inputs hook function.

        :parameter x: tensor to register in hook.
        :return: tensor to register in hook.
        """
        with no_nncf_trace():
            x_unwrapped = maybe_get_values_from_torch_return_type(x)
            collector.register_input_for_all_reducers(PTNNCFTensor(x_unwrapped))
        return x

    return register_inputs_hook
