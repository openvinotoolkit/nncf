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

import torch.fx

from nncf.common.graph.transformations.commands import Command
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.commands import TransformationType


class FXApplyTransformationCommand(Command):
    """
    Command to apply given transformation to a model.
    """

    def __init__(
        self,
        transformation_fn: Callable[[torch.fx.GraphModule], None],
        priority: Union[TransformationPriority, int] = TransformationPriority.DEFAULT_PRIORITY,
    ):
        """
        :param transformation_fn: Target transformation function.
        :param priority: Transformation priority.
        """
        super().__init__(TransformationType.INSERT)
        self.transformation_fn = transformation_fn
        self.priority = priority
