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

from typing import Callable

from torch import nn

from nncf.common.graph import NNCFNodeName
from nncf.common.pruning.structs import PrunedLayerInfoBase
from nncf.torch.dynamic_graph.scope import Scope


class PrunedModuleInfo(PrunedLayerInfoBase):
    def __init__(
        self,
        node_name: NNCFNodeName,
        module_scope: Scope,
        module: nn.Module,
        operand: Callable,
        node_id: int,
        is_depthwise: bool,
    ):
        super().__init__(node_name, node_id, is_depthwise)
        self.module_scope = module_scope
        self.module = module
        self.operand = operand
        self.key = self.node_name
