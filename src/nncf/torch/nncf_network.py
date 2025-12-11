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

from enum import IntEnum
from typing import TypeVar

from torch import nn

import nncf
from nncf.common.graph.transformations.commands import TargetType
from nncf.torch.dynamic_graph.operation_address import OperationAddress

Module = TypeVar("Module", bound=nn.Module)


class PTInsertionType(IntEnum):
    NNCF_MODULE_PRE_OP = 0
    NNCF_MODULE_POST_OP = 1
    OPERATOR_PRE_HOOK = 2
    OPERATOR_POST_HOOK = 3


TARGET_TYPE_VS_PT_INSERTION_TYPE_DICT_FOR_REPLACED_MODULES = {
    TargetType.PRE_LAYER_OPERATION: PTInsertionType.NNCF_MODULE_PRE_OP,
    TargetType.POST_LAYER_OPERATION: PTInsertionType.NNCF_MODULE_POST_OP,
    TargetType.OPERATION_WITH_WEIGHTS: PTInsertionType.NNCF_MODULE_PRE_OP,
    TargetType.OPERATOR_PRE_HOOK: PTInsertionType.OPERATOR_PRE_HOOK,
    TargetType.OPERATOR_POST_HOOK: PTInsertionType.OPERATOR_POST_HOOK,
}

TARGET_TYPE_VS_PT_INSERTION_TYPE_DICT_FOR_NOT_REPLACED_MODULES = {
    TargetType.PRE_LAYER_OPERATION: PTInsertionType.OPERATOR_PRE_HOOK,
    TargetType.POST_LAYER_OPERATION: PTInsertionType.OPERATOR_POST_HOOK,
    TargetType.OPERATION_WITH_WEIGHTS: PTInsertionType.OPERATOR_PRE_HOOK,
    TargetType.OPERATOR_PRE_HOOK: PTInsertionType.OPERATOR_PRE_HOOK,
    TargetType.OPERATOR_POST_HOOK: PTInsertionType.OPERATOR_POST_HOOK,
}


class PTInsertionPoint:
    def __init__(
        self,
        target_type: TargetType,
        op_address: OperationAddress,
        input_port_id: int = None,
        replaced_modules: bool = True,
    ):
        self.insertion_type = self._get_pt_insertion_type(target_type, replaced_modules)
        self.op_address = op_address
        self.module_scope = op_address.scope_in_model
        self.input_port_id = input_port_id

    @staticmethod
    def _get_pt_insertion_type(target_type: TargetType, replaced_modules: bool) -> PTInsertionType:
        map_target_types = TARGET_TYPE_VS_PT_INSERTION_TYPE_DICT_FOR_NOT_REPLACED_MODULES
        if replaced_modules:
            map_target_types = TARGET_TYPE_VS_PT_INSERTION_TYPE_DICT_FOR_REPLACED_MODULES

        if not isinstance(target_type, TargetType) or target_type not in map_target_types:
            msg = f"Unsupported target type for PyTorch: {target_type}"
            raise nncf.InternalError(msg)
        return map_target_types[target_type]

    def __eq__(self, other: "PTInsertionPoint"):
        return (
            self.insertion_type == other.insertion_type
            and self.op_address == other.op_address
            and self.module_scope == other.module_scope
            and self.input_port_id == other.input_port_id
        )

    def __str__(self):
        return " ".join([str(v) for v in self.__dict__.values()])

    def __hash__(self):
        return hash(str(self))
