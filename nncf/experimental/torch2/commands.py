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

from typing import List, Optional

from torch import nn

from nncf.common.graph.transformations.commands import Command
from nncf.common.graph.transformations.commands import TransformationType
from nncf.experimental.torch2.function_hook.hook_storage import RemovableHookHandle
from nncf.torch.graph.transformations.commands import PTTargetPoint


class PT2InsertionCommand(Command):
    """
    Insertion operation to the models.
    """

    def __init__(
        self,
        target_points: List[PTTargetPoint],
        hook_module: nn.Module,
        *,
        handle_storage: Optional[List[RemovableHookHandle]] = None,
    ):
        super().__init__(TransformationType.INSERT)
        self.target_points = target_points
        self.hook_module = hook_module
        self.handle_storage = handle_storage
