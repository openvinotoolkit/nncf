"""
 Copyright (c) 2020 Intel Corporation
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
from nncf.dynamic_graph.context import Scope
from nncf.graph.graph import InputAgnosticOperationExecutionContext


class QuantizerId:
    """ Unique identifier of a quantizer. It's used to store and search all quantizers in a single
    structure. Also it provides the scope, where the quantizer was inserted. """

    def get_base(self):
        raise NotImplementedError

    def get_suffix(self) -> str:
        raise NotImplementedError

    def get_scope(self) -> Scope:
        raise NotImplementedError

    def __str__(self):
        return str(self.get_base()) + self.get_suffix()

    def __hash__(self):
        return hash((self.get_base(), self.get_suffix()))

    def __eq__(self, other: 'QuantizerId'):
        return (self.get_base() == other.get_base()) and (self.get_suffix() == other.get_suffix())


class WeightQuantizerId(QuantizerId):
    """ Unique identifier of a quantizer for weights."""

    def __init__(self, scope: 'Scope'):
        self.scope = scope

    def get_base(self) -> 'Scope':
        return self.scope

    def get_suffix(self) -> str:
        return 'module_weight'

    def get_scope(self) -> Scope:
        return self.get_base()


class NonWeightQuantizerId(QuantizerId):
    """ Unique identifier of a quantizer, which corresponds to non-weight operations, such as
    ordinary activation, function and input"""

    def __init__(self, ia_op_exec_context: InputAgnosticOperationExecutionContext,
                 input_port_id=None):
        self.ia_op_exec_context = ia_op_exec_context
        self.input_port_id = input_port_id

    def get_base(self) -> 'InputAgnosticOperationExecutionContext':
        return self.ia_op_exec_context

    def get_suffix(self) -> str:
        return '|OUTPUT' if self.input_port_id is None else '|INPUT{}'.format(self.input_port_id)

    def get_scope(self) -> Scope:
        return self.ia_op_exec_context.scope_in_model


class InputQuantizerId(NonWeightQuantizerId):
    """ Unique identifier of a quantizer for model's input"""

    def get_base(self) -> 'Scope':
        return self.ia_op_exec_context.scope_in_model

    def get_suffix(self) -> str:
        return 'module_input'


class FunctionQuantizerId(NonWeightQuantizerId):
    """ Unique identifier of a quantizer for a function call"""

    def __init__(self, ia_op_exec_context: InputAgnosticOperationExecutionContext, input_arg_idx: int):
        super().__init__(ia_op_exec_context)
        self.input_arg_idx = input_arg_idx

    def get_suffix(self) -> str:
        return "_input" + str(self.input_arg_idx)
