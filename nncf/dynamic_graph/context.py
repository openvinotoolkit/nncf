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

import re
import threading
from collections import OrderedDict
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from itertools import islice
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import torch

from nncf.debug import is_debug
from nncf.dynamic_graph.graph import DynamicGraph
from nncf.graph.graph import InputAgnosticOperationExecutionContext
from nncf.graph.graph import PTNNCFNode
from nncf.graph.graph import NNCFNode
from nncf.dynamic_graph.trace_tensor import TensorMeta
from nncf.graph.version_agnostic_op_names import get_version_agnostic_name
from nncf.layers import ITERATION_MODULES
from nncf.graph.graph import ModuleAttributes
from nncf.utils import maybe_get_iterator

_CURRENT_CONTEXT = None


def nth(iterable, n, default=None):
    return next(islice(iterable, n, None), default)


class InputIndexEntry:
    def __init__(self, path: Tuple[Union[int, str], ...], getter: Callable, setter: Callable):
        self.path = path
        self.getter = getter
        self.setter = setter


class TupleRebuildingSetter:
    def __init__(self, idx_to_set, current_tuple, previous_level_setter_for_current_tuple):
        self._previous_level_setter = previous_level_setter_for_current_tuple
        self._current_tuple = current_tuple
        self._idx_to_set = idx_to_set

    def __call__(self, value):
        tmp_list = list(self._current_tuple)
        tmp_list[self._idx_to_set] = value
        new_tuple = tuple(tmp_list)
        self._current_tuple = new_tuple
        self._previous_level_setter(new_tuple)


class OperatorInput:
    def __init__(self, op_args, op_kwargs):
        self.op_args = op_args
        self.op_kwargs = op_kwargs
        self._index = OrderedDict()  # type: Dict[int, InputIndexEntry]

        op_args_index_entries = []
        self._nested_object_paths_generator(self.op_args, op_args_index_entries,
                                            previous_level_setter=partial(setattr, self, "op_args"))
        op_kwargs_index_entries = []
        self._nested_object_paths_generator(self.op_kwargs, op_kwargs_index_entries)

        # pylint:disable=unnecessary-comprehension
        self._index = {idx: entry for idx, entry in
                       enumerate(op_args_index_entries + op_kwargs_index_entries)}

    @staticmethod
    def _nested_object_paths_generator(obj, out_entries_list, path=(), memo=None, previous_level_setter=None):
        if memo is None:
            memo = set()
        iterator = maybe_get_iterator(obj)
        if iterator is not None:
            if id(obj) not in memo:
                memo.add(id(obj))
                current_level_getters = []
                current_level_setters = []
                for idx, iterval in enumerate(iterator(obj)):
                    path_component, value = iterval
                    current_level_getters.append(partial(obj.__getitem__, path_component))
                    if not isinstance(obj, tuple):
                        current_level_setters.append(partial(obj.__setitem__, path_component))
                    else:
                        current_level_setters.append(TupleRebuildingSetter(idx, obj, previous_level_setter))

                for idx, iterval in enumerate(iterator(obj)):
                    path_component, value = iterval
                    retval = OperatorInput._nested_object_paths_generator(value, out_entries_list,
                                                                          path + (path_component,), memo,
                                                                          current_level_setters[idx])
                    was_leaf = retval[1]
                    if was_leaf:
                        leaf_entry_path = retval
                        # getter = partial(obj.__getitem__, path_component)
                        getter = current_level_getters[idx]
                        setter = current_level_setters[idx]

                        out_entries_list.append(InputIndexEntry(leaf_entry_path,
                                                                getter,
                                                                setter))

                memo.remove(id(obj))
            is_leaf = False
            return path, is_leaf

        is_leaf = True
        return path, is_leaf

    def __iter__(self):
        return iter(self._index.values())

    def __getitem__(self, n):
        return self._index[n].getter()

    def __setitem__(self, n, value):
        self._index[n].setter(value)

    def __len__(self):
        return len(self._index)


class ScopeElement:
    def __init__(self, calling_module_class_name: str, calling_field_name: str = None):
        self.calling_module_class_name = calling_module_class_name
        self.calling_field_name = calling_field_name

    def __str__(self):
        if self.calling_field_name is None:
            return self.calling_module_class_name
        return "{cls}[{name}]".format(cls=self.calling_module_class_name,
                                      name=self.calling_field_name)

    def __eq__(self, other: 'ScopeElement'):
        return (self.calling_module_class_name == other.calling_module_class_name) and \
               (self.calling_field_name == other.calling_field_name)

    def __hash__(self):
        return hash((self.calling_module_class_name, self.calling_field_name))

    @staticmethod
    def from_str(string: str):
        matches = re.search(r"(.*)\[(.*)\]|(.*)", string)
        if matches is None:
            raise RuntimeError("Invalid scope element string")
        if matches.groups()[0] is None and matches.groups()[1] is None:
            return ScopeElement(matches.groups()[2])
        if matches.groups()[0] is not None and matches.groups()[1] is not None:
            return ScopeElement(matches.groups()[0], matches.groups()[1])
        raise RuntimeError("Could not parse the scope element string")


class Scope:
    def __init__(self, scope_elements: List[ScopeElement] = None):
        if scope_elements is not None:
            self.scope_elements = scope_elements
        else:
            self.scope_elements = []

    def __str__(self):
        return '/'.join([str(scope_el) for scope_el in self.scope_elements])

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: 'Scope'):
        return self.scope_elements == other.scope_elements

    def __getitem__(self, key):
        return self.scope_elements[key]

    def __contains__(self, item: 'Scope'):
        """Idiom: ('A/B/C' in 'A/B') == True"""
        if len(self.scope_elements) > len(item.scope_elements):
            return False
        for i in range(len(self.scope_elements)):
            if self.scope_elements[i] != item.scope_elements[i]:
                return False
        return True

    def __add__(self, rhs):
        init_list = self.scope_elements + rhs.scope_elements
        return Scope(init_list)

    def copy(self):
        return Scope(deepcopy(self.scope_elements))

    def push(self, scope_element: ScopeElement):
        self.scope_elements.append(scope_element)

    def pop(self) -> ScopeElement:
        return self.scope_elements.pop()

    @staticmethod
    def from_str(string: str) -> 'Scope':
        if string:
            elts = string.split('/')
        else:
            elts = []
        return Scope([ScopeElement.from_str(s) for s in elts])

    def get_iteration_scopes(self) -> List[str]:
        results = []
        scope_name = str(self)
        for iter_scope in ITERATION_MODULES.registry_dict:
            if iter_scope in scope_name:
                results.append(iter_scope)
        return results


class PreHookId:
    def __init__(self, ia_op_exec_context: InputAgnosticOperationExecutionContext,
                 input_port_id: int):
        self.ia_op_exec_context = ia_op_exec_context
        self.input_port_id = input_port_id

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self.ia_op_exec_context) + "|INPUT{}".format(self.input_port_id)

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
        self.is_tracing = True
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
                           ia_op_exec_context: InputAgnosticOperationExecutionContext) -> Optional[PTNNCFNode]:
        with self._cond:
            self._n_instances_searching_graph += 1

        node = self.graph.find_node(ia_op_exec_context, tensor_metas, self._input_comparators_per_scope)

        with self._cond:
            self._n_instances_searching_graph -= 1
            self._cond.notify_all()
        return node

    def maybe_add_node(self, inputs: OperatorInput, tensor_metas: List[Optional[TensorMeta]],
                       ia_op_exec_context: InputAgnosticOperationExecutionContext,
                       module_attrs: ModuleAttributes = None) -> NNCFNode:
        if not self._may_add_nodes:
            return None
        with self._cond:
            while self._n_instances_searching_graph > 0:
                self._cond.wait()
            # Another thread may have added a node inside this block,
            # so we need to check again if a node is already added.
            node = self.graph.find_node(ia_op_exec_context, tensor_metas, self._input_comparators_per_scope)
            if node is None:
                node = self.graph.add_node(ia_op_exec_context, tensor_metas, self._input_comparators_per_scope,
                                           inputs, module_attrs)
        return node

    def get_caller_context(self, operator_type: str) -> InputAgnosticOperationExecutionContext:
        """
        Designed to work in the following way - for each scope the context will track the number of the calls to the
        operators with the name operator_type (call_order). The counter values are preserved until reset by a
        corresponding member function of the context, which must be called after each model iteration - this is
        usually handled inside NNCF. This mechanism allows to discern between multiple function calls inside the same
        module that would each require their own instance of compression layers - for instance, multiple `relu`
        function calls (either on their own or inside a `for` cycle), and at the same moment allow the checkpoints to
        be loaded if the model had changed in the meantime in a way that does not impact the major function call
        order (e.g. if comments were added to the .py file with the model)
        """
        version_agnostic_operator_type = get_version_agnostic_name(operator_type)

        call_order = self.get_operator_call_count_in_scope(version_agnostic_operator_type, self.scope)

        ia_op_exec_context = InputAgnosticOperationExecutionContext(version_agnostic_operator_type,
                                                                    self.scope,
                                                                    call_order)
        return ia_op_exec_context

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

    def register_pre_hooks(self, fn_list: List[Callable], ia_op_exec_context: InputAgnosticOperationExecutionContext,
                           input_port_id: int):
        pre_hook_id = PreHookId(ia_op_exec_context, input_port_id)
        if pre_hook_id in self._pre_hooks:
            raise KeyError("Pre hook for context {} is already registered".format(str(pre_hook_id)))
        self._pre_hooks[pre_hook_id] = fn_list

    def execute_pre_hooks(self, ia_op_exec_context: InputAgnosticOperationExecutionContext,
                          op_inputs: OperatorInput) -> OperatorInput:
        in_op = getattr(self, 'in_operator', False)
        self.in_operator = False
        self._thread_local.num_nested_hooks += 1

        pre_hook_ids_for_curr_op = [x for x in self._pre_hooks if x.ia_op_exec_context == ia_op_exec_context]
        pre_hook_ids_for_curr_op = sorted(pre_hook_ids_for_curr_op, key=lambda x: x.input_port_id)
        for pre_hook_id in pre_hook_ids_for_curr_op:
            hook_list_for_current_input_port = self._pre_hooks[pre_hook_id]
            input_arg_to_process = pre_hook_id.input_port_id
            for hook in hook_list_for_current_input_port:
                op_inputs[input_arg_to_process] = hook(op_inputs[input_arg_to_process])
        self._thread_local.num_nested_hooks -= 1
        self.in_operator = in_op
        return op_inputs

    def register_post_hooks(self, fn_list: List[Callable], ia_op_exec_context: InputAgnosticOperationExecutionContext):
        if ia_op_exec_context in self._post_hooks:
            raise KeyError("Post hook for context {} is already registered".format(str(ia_op_exec_context)))
        self._post_hooks[ia_op_exec_context] = fn_list

    def execute_post_hooks(self, ia_op_exec_context: InputAgnosticOperationExecutionContext, outputs):
        in_op = getattr(self, 'in_operator', False)
        self.in_operator = False
        self._thread_local.num_nested_hooks += 1
        if ia_op_exec_context in self._post_hooks:
            for hook in self._post_hooks[ia_op_exec_context]:
                outputs = hook(outputs)
        self._thread_local.num_nested_hooks -= 1
        self.in_operator = in_op
        return outputs

    def disable_tracing(self):
        self.is_tracing = False

    def enable_tracing(self):
        self.is_tracing = True

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

    def register_node_call(self, node_key: str):
        if node_key in self._thread_local.node_call_tracker:
            self._thread_local.node_call_tracker[node_key] += 1
        else:
            self._thread_local.node_call_tracker[node_key] = 1

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


def get_current_context() -> TracingContext:
    return _CURRENT_CONTEXT
