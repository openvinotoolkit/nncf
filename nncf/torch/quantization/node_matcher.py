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
from nncf.common.graph import NNCFNode

from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES
from nncf.torch.graph.operator_metatypes import PTOperatorMetatype


class PTOperatorMetatypeNodeMatcher:
    @classmethod
    def match(cls, nncf_node: NNCFNode) -> PTOperatorMetatype:
        layer_attributes = nncf_node.layer_attributes
        op_name = nncf_node.node_type
        op_arch = PT_OPERATOR_METATYPES.get_operator_metatype_by_op_name(op_name)
        if op_arch.subtypes:
            subtype = op_arch.determine_subtype(layer_attributes=layer_attributes)
            if subtype is not None:
                op_arch = subtype
        return op_arch
