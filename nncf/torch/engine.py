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

from typing import Dict, Tuple, Union

import torch
from torch import nn

from nncf import nncf_logger
from nncf.common.engine import Engine
from nncf.experimental.tensor import Tensor


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

    def infer(self, input_data: Union[torch.Tensor, Tuple[torch.Tensor], Dict[str, torch.Tensor]]) -> Dict[str, Tensor]:
        """
        Runs Torch model on the provided input.

        :param input_data: Inputs for the model.
        :return: Model outputs.
        """
        output = self._model(input_data)
        if isinstance(output, torch.Tensor):
            return {None: Tensor(output)}
        if isinstance(output, dict):
            output_dict = {}
            for key, value in output.items():
                if not isinstance(value, torch.Tensor):
                    nncf_logger.debug(
                        f"PTEngine: model output dict has non-tensor value {value} for key {key}, "
                        f"will skip this value when considering tensor outputs"
                    )
                    continue
                output_dict[key] = Tensor(value)
            return output_dict

        if isinstance(input_data, dict):
            return self._model(**input_data)
        if isinstance(input_data, tuple):
            return self._model(*input_data)
        return self._model(input_data)
