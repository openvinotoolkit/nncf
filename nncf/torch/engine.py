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

from typing import Any, Dict, Union

import torch
from torch import nn

from nncf.common.engine import Engine


class PTEngine(Engine):
    """
    Engine for the Pytorch backend.
    """

    def __init__(self, model: nn.Module):
        """
        Constructor.

        :param model: Pytorch module to infer.
        """

        self._model = model
        self._model.eval()

    def infer(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Runs Torch model on the provided input.

        :param input_data: inputs for the model
        :return output_data: model outputs
        """

        return self._model(input_data)
