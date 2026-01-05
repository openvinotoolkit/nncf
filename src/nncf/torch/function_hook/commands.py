# Copyright (c) 2026 Intel Corporation
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

import torch
from torch import nn

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import Command
from nncf.common.graph.transformations.commands import TransformationType
from nncf.torch.function_hook.hook_storage import RemovableHookHandle
from nncf.torch.graph.transformations.commands import PTTargetPoint


class PT2InsertionCommand(Command):
    """
    Insertion operation to the models.
    """

    def __init__(
        self,
        target_points: list[PTTargetPoint],
        hook_module: nn.Module,
        *,
        handle_storage: Optional[list[RemovableHookHandle]] = None,
    ):
        """
        :param target_points: The list of target points for the command.
        :param hook_module: The hook module for the command that will be inserted into the model
          to execute at the target points.
        :param handle_storage: The handle storage for the command to collect RemovableHookHandle. Defaults to None.
        """
        super().__init__(TransformationType.INSERT)
        self.target_points = target_points
        self.hook_module = hook_module
        self.handle_storage = handle_storage


class PT2ConstUpdateCommand(Command):
    """
    Corrects weight value in the model based on the input value.
    """

    def __init__(self, node: NNCFNode, value: torch.Tensor):
        """
        :param const_node: The node of the data in the model.
        :param value: The new value of the constant.
        """
        super().__init__(TransformationType.CHANGE)
        self.node = node
        self.value = value
