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
from nncf.graph.graph import PTNNCFNode
from nncf.graph.operator_metatypes import OPERATOR_METATYPES
from nncf.graph.operator_metatypes import OperatorMetatype


class PTOperatorMetatypeNodeMatcher:
    @classmethod
    def match(cls, nncf_node: PTNNCFNode) -> OperatorMetatype:
        ia_op_exec_context = nncf_node.ia_op_exec_context
        module_attributes = nncf_node.module_attributes
        op_name = ia_op_exec_context.operator_name
        op_arch = OPERATOR_METATYPES.get_operator_metatype_by_op_name(op_name)
        if op_arch.subtypes:
            subtype = op_arch.determine_subtype(module_attributes=module_attributes)
            if subtype is not None:
                op_arch = subtype
        return op_arch
