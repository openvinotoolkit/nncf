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

import threading
import weakref
from collections import OrderedDict
from collections import defaultdict
from collections import deque
from contextlib import contextmanager
from typing import Callable, DefaultDict, Dict, List, Optional, Union

import torch

from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.hook_handle import HookHandle
from nncf.common.hook_handle import add_op_to_registry
from nncf.common.utils.api_marker import api
from nncf.common.utils.debug import is_debug
from nncf.common.utils.patcher import PATCHER
from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.dynamic_graph.graph import DynamicGraphNode
from nncf.torch.dynamic_graph.graph import DynamicGraphNodeParameters
from nncf.torch.dynamic_graph.graph import TensorMetaComparator
from nncf.torch.dynamic_graph.op_input_processing import OperatorInput
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.dynamic_graph.scope import ScopeElement
from nncf.torch.dynamic_graph.trace_tensor import TensorMeta
from nncf.torch.dynamic_graph.trace_tensor import TracedTensorMixin


class ThreadLocalGlobalContext(threading.local):
    def __init__(self):
        super().__init__()
        self.context = None


_CURRENT_CONTEXT = ThreadLocalGlobalContext()


class PreHookId:
    def __init__(self, op_address: OperationAddress, input_port_id: int):
        self.op_address = op_address
        self.input_port_id = input_port_id

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self.op_address) + "|INPUT{}".format(self.input_port_id)

    def __hash__(self):
        return hash(str(self))


class TracingThreadLocals(threading.local):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.scopes = []
        self.module_call_stack = []
        self.in_operator = False
        self.in_parameter_trace = False
        self.num_nested_hooks = 0
        self.base_module_replica = None
        self.operator_counters = {}
        self.node_call_tracker = {}
        self.traced_tensor_weakrefs = []
        self.nested_contexts_stack = []
        self.processed_parameters = {}


class CopySafeThreadingVars:
    """A class holding variables that are related to threading and
    thus impossible to deepcopy. The deepcopy will simply return a
    new object without copying, but won't fail."""

    def __init__(self):
        self.thread_local = TracingThreadLocals()
        self.cond = threading.Condition()

    def __deepcopy__(self, memo):
        return CopySafeThreadingVars()


class TracingContext:
    def __init__(self):
        self.graph = DynamicGraph()

        self._post_hooks: DefaultDict[OperationAddress, Dict[str, Callable]] = defaultdict(OrderedDict)
        self._pre_hooks: DefaultDict[PreHookId, Dict[str, Callable]] = defaultdict(OrderedDict)
        self._num_nested_hooks = 0
        self.reused_parameters = []

        self._threading = CopySafeThreadingVars()

        self._n_instances_searching_graph = 0

        self._is_tracing = True
        self._is_forwarding = False
        self._may_add_nodes = True
        self._input_comparators_per_scope = []
        self.global_buffer_store = {}

        self._trace_dynamic_graph = False

        # ELASTIC DEPTH PARAMS
        self.elastic_depth = False
        self.skipped_blocks = []
        self.in_skipped_block = False
        self.start_node_name_of_skipped_block = []
        self.end_node_name_of_skipped_block = []
        self.active_block_indexes = None
        self.tensor_cache = None
        self._ordinals_ids = None

    def __enter__(self):
        # For DataParallel, this relies on having the same compressed context for
        # all replicas. Otherwise we will have data races on setting and reading the global _CURRENT_CONTEXT
        # variable, which will in turn lead to DP-specific runtime errors such as
        # "'_thread._local' object has no attribute 'scopes'"
        self._threading.thread_local.nested_contexts_stack.append(get_current_context())
        set_current_context(self)

        return self

    def __exit__(self, *args):
        previous_context = self._threading.thread_local.nested_contexts_stack.pop(-1)
        for traced_tensor_weakref in self._threading.thread_local.traced_tensor_weakrefs:
            tt = traced_tensor_weakref()
            if tt is None or not isinstance(tt, TracedTensorMixin):
                continue
            if previous_context is None:
                tt.strip()
            elif previous_context is not self:
                previous_context.register_traced_tensor(tt)

        if previous_context is not self:
            self._reset_thread_local()

            if is_debug():
                self.reset_node_call_counters()

        set_current_context(previous_context)

    def find_operator_node(
        self, tensor_metas: List[Optional[TensorMeta]], op_address: OperationAddress
    ) -> Optional[DynamicGraphNode]:
        with self._threading.cond:
            self._n_instances_searching_graph += 1

        node = self.graph.find_node(op_address, tensor_metas, self._input_comparators_per_scope)

        with self._threading.cond:
            self._n_instances_searching_graph -= 1
            self._threading.cond.notify_all()
        return node

    def register_global_buffer(self, name: str, buffer):
        self.global_buffer_store[name] = buffer

    def register_traced_tensor(self, tt: torch.Tensor):
        """
        Registers a weak reference to a traced tensor in the context so that in case
        the block under context retains a reference to an intermediate tensor somewhere,
        the context strips this traced tensor upon context exit.

        :param tt: A tensor with TracedTensorMixin tracing capabilities to be registered.
        """
        wr = weakref.ref(tt)
        self._threading.thread_local.traced_tensor_weakrefs.append(wr)

    def register_processed_parameter(self, param_name: str, tensor: torch.Tensor) -> None:
        """
        Registers the processed parameter in the context to avoid double calculation of hooks
        for the same parameters.

        :param param_name: The parameter name.
        :param tensor: The processed parameter.
        """
        self._threading.thread_local.processed_parameters[param_name] = tensor

    def get_processed_parameter(self, param_name: str) -> Union[torch.Tensor, None]:
        """
        Return the processed parameter by name.

        :param param_name: The parameter name.
        :return: The processed parameter by name if found, otherwise None.
        """
        return self._threading.thread_local.processed_parameters.get(param_name, None)

    def maybe_add_node(
        self,
        inputs: OperatorInput,
        tensor_metas: List[Optional[TensorMeta]],
        op_address: OperationAddress,
        module_attrs: BaseLayerAttributes = None,
        ignored_algorithms: List[str] = None,
        is_called_inside_nncf_module: bool = False,
    ) -> Optional[DynamicGraphNode]:
        if not self._may_add_nodes:
            return None
        with self._threading.cond:
            while self._n_instances_searching_graph > 0:
                self._threading.cond.wait()
            # Another thread may have added a node inside this block,
            # so we need to check again if a node is already added.
            node = self.graph.find_node(op_address, tensor_metas, self._input_comparators_per_scope)
            if node is None:
                mid = id(self.get_current_module())
                node = self.graph.add_node(
                    op_address,
                    tensor_metas,
                    self._input_comparators_per_scope,
                    inputs,
                    DynamicGraphNodeParameters(
                        module_attrs, ignored_algorithms, is_called_inside_nncf_module, calling_module_id=mid
                    ),
                )
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

        op_address = OperationAddress(operator_name, self.scope, call_order)
        return op_address

    def reset_scope_operator_call_counters(self):
        """
        Must be called after each "forward" operation of the model that is made
        within this context
        """
        self._threading.thread_local.operator_counters = {}

    @staticmethod
    def _get_operator_counter_key(operator_name: str, scope: Scope):
        return "{}_{}".format(str(scope), operator_name)

    def register_operator_call(self, operator_name: str, scope: Scope):
        key = self._get_operator_counter_key(operator_name, scope)
        if key in self._threading.thread_local.operator_counters:
            self._threading.thread_local.operator_counters[key] += 1
        else:
            self._threading.thread_local.operator_counters[key] = 1

    def get_operator_call_count_in_scope(self, operator_name: str, scope: Scope):
        key = self._get_operator_counter_key(operator_name, scope)
        if key in self._threading.thread_local.operator_counters:
            return self._threading.thread_local.operator_counters[key]
        return 0

    def reset_operator_call_count_in_scope(self, scope):
        scoped_op_name = str(scope)
        for key in self._threading.thread_local.operator_counters:
            if scoped_op_name in key:
                self._threading.thread_local.operator_counters[key] = 0

    def push_scope(self, called_module: torch.nn.Module):
        relative_scopes_list = self._get_scope_relative_to_last_registered_module_call(called_module)
        self.module_call_stack.append(called_module)
        self.relative_scopes_stack.append(relative_scopes_list)

    def pop_scope(self):
        self.relative_scopes_stack.pop()
        self.module_call_stack.pop()

    def register_pre_hook(self, fn: Callable, op_address: OperationAddress, input_port_id: int) -> HookHandle:
        pre_hook_id = PreHookId(op_address, input_port_id)
        return add_op_to_registry(self._pre_hooks[pre_hook_id], fn)

    def execute_pre_hooks(self, op_address: OperationAddress, op_inputs: OperatorInput) -> OperatorInput:
        in_op = getattr(self, "in_operator", False)
        self.in_operator = False
        self._threading.thread_local.num_nested_hooks += 1

        pre_hook_ids_for_curr_op = [x for x in self._pre_hooks if x.op_address == op_address]
        pre_hook_ids_for_curr_op = sorted(pre_hook_ids_for_curr_op, key=lambda x: x.input_port_id)
        for pre_hook_id in pre_hook_ids_for_curr_op:
            hook_list_for_current_input_port = self._pre_hooks[pre_hook_id].values()
            input_arg_to_process = pre_hook_id.input_port_id
            for hook in hook_list_for_current_input_port:
                op_inputs[input_arg_to_process] = hook(op_inputs[input_arg_to_process])
        self._threading.thread_local.num_nested_hooks -= 1
        self.in_operator = in_op
        return op_inputs

    def register_post_hook(self, fn: Callable, op_address: OperationAddress) -> HookHandle:
        return add_op_to_registry(self._post_hooks[op_address], fn)

    def execute_post_hooks(self, op_address: OperationAddress, outputs):
        in_op = getattr(self, "in_operator", False)
        self.in_operator = False
        self._threading.thread_local.num_nested_hooks += 1
        if op_address in self._post_hooks:
            for hook in self._post_hooks[op_address].values():
                outputs = hook(outputs)
        self._threading.thread_local.num_nested_hooks -= 1
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

    def add_node_comparators(self, scopes_to_apply: List[str], node_input_comparator: TensorMetaComparator = None):
        self._input_comparators_per_scope.append((node_input_comparator, scopes_to_apply))

    @property
    def base_module_thread_local_replica(self) -> torch.nn.Module:
        return self._threading.thread_local.base_module_replica

    @base_module_thread_local_replica.setter
    def base_module_thread_local_replica(self, value: torch.nn.Module):
        self._threading.thread_local.base_module_replica = value

    @property
    def in_operator(self):
        return self._threading.thread_local.in_operator

    @in_operator.setter
    def in_operator(self, val):
        self._threading.thread_local.in_operator = val

    @property
    def in_parameter_trace(self):
        return self._threading.thread_local.in_parameter_trace

    @in_parameter_trace.setter
    def in_parameter_trace(self, val):
        self._threading.thread_local.in_parameter_trace = val

    @property
    def module_call_stack(self) -> List[torch.nn.Module]:
        return self._threading.thread_local.module_call_stack

    def get_current_module(self) -> Optional[torch.nn.Module]:
        if self.module_call_stack:
            return self.module_call_stack[-1]
        return None

    @property
    def relative_scopes_stack(self) -> List[Scope]:
        return self._threading.thread_local.scopes

    @property
    def trace_dynamic_graph(self) -> bool:
        return self._trace_dynamic_graph

    def disable_trace_dynamic_graph(self):
        if not self.graph.is_graph_with_iteration_modules():
            self._trace_dynamic_graph = False

    def enable_trace_dynamic_graph(self):
        self._trace_dynamic_graph = True

    def _reset_thread_local(self):
        self._threading.thread_local.reset()

    def register_node_call(self, node: DynamicGraphNode):
        if node.node_id in self._threading.thread_local.node_call_tracker:
            self._threading.thread_local.node_call_tracker[node.node_id] += 1
        else:
            self._threading.thread_local.node_call_tracker[node.node_id] = 1

    def reset_node_call_counters(self):
        for k, _ in self._threading.thread_local.node_call_tracker.items():
            self._threading.thread_local.node_call_tracker[k] = 0

    def get_node_call_counter_dict(self):
        return self._threading.thread_local.node_call_tracker

    def _get_scope_relative_to_last_registered_module_call(self, module) -> Scope:
        module_class = module.__class__.__name__
        if not self.module_call_stack:
            return Scope(
                [
                    ScopeElement(module_class),
                ]
            )
        q = deque([(tuple(), self.module_call_stack[-1])])
        while q:
            scope_parts, top = q.popleft()
            if module is top:
                return Scope(list(scope_parts))
            for name, child in top.named_children():
                scope_element = ScopeElement(child.__class__.__name__, name)
                q.append((scope_parts + (scope_element,), child))
        return Scope(
            [
                ScopeElement(module_class),
            ]
        )

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

    def set_active_skipped_block(self, block_indexes: List[int]):
        if self.active_block_indexes is not None:
            self.start_node_name_of_skipped_block = []
            self.end_node_name_of_skipped_block = []
        self.active_block_indexes = block_indexes
        if self.elastic_depth and len(block_indexes) > 0:
            for block_index in block_indexes:
                self.start_node_name_of_skipped_block.append(self.skipped_blocks[block_index].start_node_name)
                self.end_node_name_of_skipped_block.append(self.skipped_blocks[block_index].end_node_name)

    def set_elastic_blocks(self, blocks: List["BuildingBlock"] = None):  # noqa: F821
        if blocks is not None and isinstance(blocks, list):
            if len(blocks) == 0:
                self.skipped_blocks = []
            elif isinstance(blocks[0], str):
                self.skipped_blocks = [blocks]
            else:
                self.skipped_blocks = blocks


def set_current_context(c: TracingContext):
    _CURRENT_CONTEXT.context = c


@api(canonical_alias="nncf.torch.no_nncf_trace")
@contextmanager
def no_nncf_trace():
    ctx = get_current_context()
    if ctx is not None and ctx.is_tracing:
        ctx.disable_tracing()
        yield
        ctx.enable_tracing()
    else:
        yield


@api(canonical_alias="nncf.torch.forward_nncf_trace")
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
    return _CURRENT_CONTEXT.context


@api(canonical_alias="nncf.torch.disable_tracing")
def disable_tracing(method):
    """
    Patch a method so that it will be executed within no_nncf_trace context
    :param method: A method to patch.
    """

    def no_nncf_trace_wrapper(self, fn, *args, **kwargs):
        with no_nncf_trace():
            return fn(*args, **kwargs)

    PATCHER.patch(method, no_nncf_trace_wrapper)
