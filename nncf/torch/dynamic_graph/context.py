"""
 Copyright (c) 2019-2020 Intel Corporation
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

import threading
from collections import deque
from contextlib import contextmanager
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import torch

from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.torch.debug import is_debug
from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.dynamic_graph.graph import DynamicGraphNode
from nncf.torch.dynamic_graph.op_input_processing import OperatorInput
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.dynamic_graph.scope import ScopeElement
from nncf.torch.dynamic_graph.trace_tensor import TensorMeta

_CURRENT_CONTEXT = None


class PreHookId:
    def __init__(self, op_address: OperationAddress,
                 input_port_id: int):
        self.op_address = op_address
        self.input_port_id = input_port_id

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self.op_address) + "|INPUT{}".format(self.input_port_id)

    def __hash__(self):
        return hash(str(self))


# pylint: disable=too-many-public-methods
class TracingContext:
    def __init__(self):
        self.graph = DynamicGraph()

        self._save_context = None
        self._post_hooks = {}
        self._pre_hooks = {}  # type: Dict[PreHookId, List[Callable]]
        self._num_nested_hooks = 0

        self._thread_local = threading.local()

        self._n_instances_searching_graph = 0
        self._cond = threading.Condition()
        self._is_tracing = True
        self._is_forwarding = False
        self._may_add_nodes = True
        self._input_comparators_per_scope = []

    def __enter__(self):
        global _CURRENT_CONTEXT
        self._save_context = _CURRENT_CONTEXT
        _CURRENT_CONTEXT = self
        self._init_thread_local()
        if is_debug():
            self.reset_node_call_counters()

        return self

    def __exit__(self, *args):
        self.reset_scope_operator_call_counters()
        self.leave()

    def find_operator_node(self, tensor_metas: List[Optional[TensorMeta]],
                           op_address: OperationAddress) -> Optional[DynamicGraphNode]:
        with self._cond:
            self._n_instances_searching_graph += 1

        node = self.graph.find_node(op_address, tensor_metas, self._input_comparators_per_scope)

        with self._cond:
            self._n_instances_searching_graph -= 1
            self._cond.notify_all()
        return node

    def maybe_add_node(self, inputs: OperatorInput, tensor_metas: List[Optional[TensorMeta]],
                       op_address: OperationAddress,
                       module_attrs: BaseLayerAttributes = None,
                       ignored_algorithms: List[str] = None) -> Optional[DynamicGraphNode]:
        if not self._may_add_nodes:
            return None
        with self._cond:
            while self._n_instances_searching_graph > 0:
                self._cond.wait()
            # Another thread may have added a node inside this block,
            # so we need to check again if a node is already added.
            node = self.graph.find_node(op_address, tensor_metas, self._input_comparators_per_scope)
            if node is None:
                node = self.graph.add_node(op_address, tensor_metas, self._input_comparators_per_scope,
                                           inputs, module_attrs, ignored_algorithms)
        return node

    def get_caller_context(self, operator_name: str) -> OperationAddress:
        """
        Designed to work in the following way - for each scope the context will track the number of the calls to the
        operators with the name operator_name (call_order). The counter values are preserved until reset by a
        corresponding member function of the context, which must be called after each model iteration - this is
        usually handled inside NNCF. This mechanism allows to discern between multiple function calls inside the same
        module that would each require their own instance of compression layers - for instance, multiple `relu`
        function calls (either on their own or inside a `for` cycle), and at the same moment allow the checkpoints to
        be loaded if the model had changed in the meantime in a way that does not impact the major function call
        order (e.g. if comments were added to the .py file with the model)
        """

        call_order = self.get_operator_call_count_in_scope(operator_name, self.scope)

        op_address = OperationAddress(operator_name,
                                      self.scope,
                                      call_order)
        return op_address

    def reset_scope_operator_call_counters(self):
        """
        Must be called after each "forward" operation of the model that is made
        within this context
        """
        self._thread_local.operator_counters = {}

    @staticmethod
    def _get_operator_counter_key(operator_name: str, scope: Scope):
        return "{}_{}".format(str(scope), operator_name)

    def register_operator_call(self, operator_name: str, scope: Scope):
        key = self._get_operator_counter_key(operator_name, scope)
        if key in self._thread_local.operator_counters:
            self._thread_local.operator_counters[key] += 1
        else:
            self._thread_local.operator_counters[key] = 1

    def get_operator_call_count_in_scope(self, operator_name: str, scope: Scope):
        key = self._get_operator_counter_key(operator_name, scope)
        if key in self._thread_local.operator_counters:
            return self._thread_local.operator_counters[key]
        return 0

    def reset_operator_call_count_in_scope(self, scope):
        scoped_op_name = str(scope)
        for key in self._thread_local.operator_counters.keys():
            if scoped_op_name in key:
                self._thread_local.operator_counters[key] = 0

    def enter(self):
        global _CURRENT_CONTEXT
        self._save_context = _CURRENT_CONTEXT
        _CURRENT_CONTEXT = self
        self._init_thread_local()

    def leave(self):
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT = self._save_context
        self._save_context = None

    def push_scope(self, called_module: torch.nn.Module):
        relative_scopes_list = self._get_scope_relative_to_last_registered_module_call(called_module)
        self.module_call_stack.append(called_module)
        self.relative_scopes_stack.append(relative_scopes_list)

    def pop_scope(self):
        self.relative_scopes_stack.pop()
        self.module_call_stack.pop()

    def register_pre_hooks(self, fn_list: List[Callable], op_address: OperationAddress,
                           input_port_id: int):
        pre_hook_id = PreHookId(op_address, input_port_id)
        if pre_hook_id in self._pre_hooks:
            raise KeyError("Pre hook for context {} is already registered".format(str(pre_hook_id)))
        self._pre_hooks[pre_hook_id] = fn_list

    def execute_pre_hooks(self, op_address: OperationAddress,
                          op_inputs: OperatorInput) -> OperatorInput:
        in_op = getattr(self, 'in_operator', False)
        self.in_operator = False
        self._thread_local.num_nested_hooks += 1

        pre_hook_ids_for_curr_op = [x for x in self._pre_hooks if x.op_address == op_address]
        pre_hook_ids_for_curr_op = sorted(pre_hook_ids_for_curr_op, key=lambda x: x.input_port_id)
        for pre_hook_id in pre_hook_ids_for_curr_op:
            hook_list_for_current_input_port = self._pre_hooks[pre_hook_id]
            input_arg_to_process = pre_hook_id.input_port_id
            for hook in hook_list_for_current_input_port:
                op_inputs[input_arg_to_process] = hook(op_inputs[input_arg_to_process])
        self._thread_local.num_nested_hooks -= 1
        self.in_operator = in_op
        return op_inputs

    def register_post_hooks(self, fn_list: List[Callable], op_address: OperationAddress):
        if op_address in self._post_hooks:
            raise KeyError("Post hook for context {} is already registered".format(str(op_address)))
        self._post_hooks[op_address] = fn_list

    def execute_post_hooks(self, op_address: OperationAddress, outputs):
        in_op = getattr(self, 'in_operator', False)
        self.in_operator = False
        self._thread_local.num_nested_hooks += 1
        if op_address in self._post_hooks:
            for hook in self._post_hooks[op_address]:
                outputs = hook(outputs)
        self._thread_local.num_nested_hooks -= 1
        self.in_operator = in_op
        return outputs

    @property
    def is_tracing(self) -> bool:
        return self._is_tracing

    def disable_tracing(self):
        self._is_tracing = False

    def enable_tracing(self):
        self._is_tracing = True

    @property
    def is_forwarding(self) -> bool:
        return self._is_forwarding

    def disable_forwarding(self):
        self._is_forwarding = False

    def enable_forwarding(self):
        self._is_forwarding = True

    def enable_node_additions(self):
        self._may_add_nodes = True

    def disable_node_additions(self):
        self._may_add_nodes = False

    def add_node_comparators(self, scopes_to_apply: List[str],
                             node_input_comparator: 'TensorMetaComparator' = None):
        self._input_comparators_per_scope.append((node_input_comparator, scopes_to_apply))

    @property
    def base_module_thread_local_replica(self):
        self._init_thread_local()
        return self._thread_local.base_module_replica

    @base_module_thread_local_replica.setter
    def base_module_thread_local_replica(self, value):
        self._init_thread_local()
        self._thread_local.base_module_replica = value

    @property
    def in_operator(self):
        self._init_thread_local()
        return self._thread_local.in_operator

    @in_operator.setter
    def in_operator(self, val):
        self._init_thread_local()
        self._thread_local.in_operator = val

    @property
    def module_call_stack(self) -> List[torch.nn.Module]:
        self._init_thread_local()
        return self._thread_local.module_call_stack

    def get_current_module(self) -> Optional[torch.nn.Module]:
        if self.module_call_stack:
            return self.module_call_stack[-1]
        return None

    @property
    def relative_scopes_stack(self) -> List[Scope]:
        self._init_thread_local()
        return self._thread_local.scopes

    def _init_thread_local(self):
        # todo: primary node part!
        tl = self._thread_local
        if getattr(tl, 'ready', False):
            return
        tl.ready = True
        tl.scopes = []
        tl.module_call_stack = []
        tl.in_operator = False
        tl.num_nested_hooks = 0
        tl.base_module_replica = None
        tl.operator_counters = {}
        tl.node_call_tracker = {}

    def register_node_call(self, node: DynamicGraphNode):
        if node.node_id in self._thread_local.node_call_tracker:
            self._thread_local.node_call_tracker[node.node_id] += 1
        else:
            self._thread_local.node_call_tracker[node.node_id] = 1

    def reset_node_call_counters(self):
        for k, _ in self._thread_local.node_call_tracker.items():
            self._thread_local.node_call_tracker[k] = 0

    def get_node_call_counter_dict(self):
        return self._thread_local.node_call_tracker

    def _get_scope_relative_to_last_registered_module_call(self, module) -> Scope:
        module_class = module.__class__.__name__
        if not self.module_call_stack:
            return Scope([ScopeElement(module_class), ])
        q = deque([(tuple(), self.module_call_stack[-1])])
        while q:
            scope_parts, top = q.popleft()
            if module is top:
                return Scope(list(scope_parts))
            for name, child in top.named_children():
                scope_element = ScopeElement(child.__class__.__name__, name)
                q.append((scope_parts + (scope_element,), child))
        return Scope([ScopeElement(module_class), ])

    @property
    def scope(self) -> Scope:
        stack_copy = self.relative_scopes_stack.copy()
        scope_el_list = []
        for relative_scope in stack_copy:
            for scope_element in relative_scope.scope_elements:
                scope_el_list.append(scope_element)
        return Scope(scope_el_list)

    def reset_graph(self):
        self.graph = DynamicGraph()


@contextmanager
def no_nncf_trace():
    ctx = get_current_context()
    if ctx is not None and ctx.is_tracing:
        ctx.disable_tracing()
        yield
        ctx.enable_tracing()
    else:
        yield

@contextmanager
def forward_nncf_trace():
    ctx = get_current_context()
    if ctx is not None and not ctx.is_forwarding:
        ctx.enable_forwarding()
        yield
        ctx.disable_forwarding()
    else:
        yield


def get_current_context() -> TracingContext:
    return _CURRENT_CONTEXT
