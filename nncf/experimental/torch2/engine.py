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

from typing import Any, Dict, Tuple, Union

import torch

from nncf.common.engine import Engine
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper


class PT2Engine(Engine):
    """
    Engine for the Pytorch backend.
    """

    def __init__(self, model: GraphModelWrapper):
        """
        Constructor.

        :param model: Pytorch module to infer.
        """

        self._model = model.model
        self._model.eval()

    def infer(self, input_data: Union[torch.Tensor, Tuple[torch.Tensor], Dict[str, torch.Tensor]]) -> Any:
        """
        Runs Torch model on the provided input.

        :param input_data: Inputs for the model.
        :return: Model outputs.
        """

        if isinstance(input_data, dict):
            return self._model(**input_data)
        if isinstance(input_data, tuple):
            return self._model(*input_data)
        return self._model(input_data)
