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
from nncf.torch.dynamic_graph.scope import Scope


class OperationAddress:
    def __init__(self, operator_name: str, scope_in_model: Scope, call_order: int):
        self.operator_name = operator_name
        self.scope_in_model = scope_in_model
        self.call_order = call_order

    def __eq__(self, other: "OperationAddress"):
        return (
            isinstance(other, OperationAddress)
            and (self.operator_name == other.operator_name)
            and (self.scope_in_model == other.scope_in_model)
            and (self.call_order == other.call_order)
        )

    def __str__(self):
        return str(self.scope_in_model) + "/" + self.operator_name + "_" + str(self.call_order)

    def __hash__(self):
        return hash((self.operator_name, self.scope_in_model, self.call_order))

    @staticmethod
    def from_str(s: str):
        scope_and_op, _, call_order_str = s.rpartition("_")
        scope_str, _, op_name = scope_and_op.rpartition("/")
        return OperationAddress(op_name, Scope.from_str(scope_str), int(call_order_str))
