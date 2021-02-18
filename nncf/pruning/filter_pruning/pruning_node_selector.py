"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from typing import List

from nncf.dynamic_graph.graph import PTNNCFNode
from nncf.common.pruning.pruning_node_selector import PruningNodeSelector
from nncf.pruning.utils import pt_is_conv_with_downsampling
from nncf.pruning.utils import pt_is_depthwise_conv
from nncf.utils import pt_should_consider_scope


class PTPruningNodeSelector(PruningNodeSelector):

    def _get_module_identifier(self, node: PTNNCFNode) -> str:
        return str(node.op_exec_context.scope_in_model)

    def _is_depthwise_conv(self, node: PTNNCFNode) -> bool:
        return pt_is_depthwise_conv(node)

    def _is_conv_with_downsampling(self, node: PTNNCFNode) -> bool:
        return pt_is_conv_with_downsampling(node)

    def _should_consider_scope(self, scope_str: str, target_scopes: List[str], ignored_scopes: List[str]):
        return pt_should_consider_scope(scope_str, target_scopes, ignored_scopes)
