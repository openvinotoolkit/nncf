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
import functools
from copy import deepcopy
from typing import Callable, List, Tuple

import torch
from torch.nn import DataParallel

from nncf.common.graph.definitions import MODEL_CONST_OP_NAME
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import WeightedLayerAttributes
from nncf.common.logging import nncf_logger
from nncf.common.utils.debug import is_debug
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.context import get_current_context
from nncf.torch.dynamic_graph.layer_attributes_handlers import get_layer_attributes_from_args_and_kwargs
from nncf.torch.dynamic_graph.op_input_processing import OperatorInput
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.dynamic_graph.patch_pytorch_state import PATCHING_STATE
from nncf.torch.dynamic_graph.structs import NamespaceTarget
from nncf.torch.dynamic_graph.structs import PatchedOperatorInfo
from nncf.torch.dynamic_graph.trace_functions import forward_trace_only
from nncf.torch.dynamic_graph.trace_functions import make_tensor_metas
from nncf.torch.dynamic_graph.trace_functions import trace_tensors
from nncf.torch.dynamic_graph.trace_tensor import TracedParameter
from nncf.torch.layer_utils import _NNCFModuleMixin
from nncf.torch.layers import ITERATION_MODULES

_IGNORED_SCOPES = []


def _warn_data_parallel():
    if getattr(_warn_data_parallel, "warned_once", False):
        return
    _warn_data_parallel.warned_once = True
    nncf_logger.warning(
        "You are using DataParallel, which may cause significant performance issues with dynamic graph "
        "building. Consider using distributed training (DistributedDataParallel) instead."
    )


def ignore_scope(cls):
    if cls not in _IGNORED_SCOPES:
        _IGNORED_SCOPES.append(cls)
    return cls


def wrap_operator(operator, operator_info: PatchedOperatorInfo):
    """
    Wraps the input callable object (`operator`) with the functionality that allows the calls to this object
    to be tracked by the currently set global TracingContext. The wrapped functions can be then intercepted,
    their arguments and return values modified arbitrarily and, for functions that correspond to operations on
    tensors in a DNN, their general position and address in the DNN's model control flow graph can be established.

    :param: operator: A callable object to be wrapped.
    :param: operator_info (PatchedOperatorInfo): An informational struct containing the specifics of wrapping
            the `operator` in question.

    :return: The wrapped version of `operator` that, without a TracingContext, performs functionally the same as
             the unwrapped version, but within a TracingContext is able to be tracked and hooked.
    """
    # do not wrap function twice
    _orig_op = getattr(operator, "_original_op", None)
    if _orig_op is not None:
        nncf_logger.debug(f"Operator: {_orig_op.__name__} is already wrapped")
        return operator

    @functools.wraps(operator)
    def wrapped(*args, **kwargs):
        if not PATCHING_STATE.operators_are_wrapped:
            # If operators are not supposed to be wrapped, skip the wrapper logic
            return operator(*args, **kwargs)

        ctx = get_current_context()
        if not ctx or getattr(ctx, "in_operator", False) or not ctx.is_tracing:
            op1 = operator(*args, **kwargs)
            return op1

        ctx.in_operator = True

        try:
            if operator_info.skip_trace:
                result = operator(*args, **kwargs)
            elif ctx.is_forwarding:
                result = forward_trace_only(operator, *args, **kwargs)
            else:
                op_name = operator_info.name
                op_address = ctx.get_caller_context(op_name)
                ctx.register_operator_call(op_address.operator_name, op_address.scope_in_model)

                if ctx.elastic_depth and ctx.in_skipped_block:
                    result = ctx.tensor_cache
                else:
                    result = _execute_op(op_address, operator_info, operator, ctx, *args, **kwargs)

                str_op_address = str(op_address)
                if str_op_address in ctx.end_node_name_of_skipped_block:
                    assert ctx.in_skipped_block is True
                    ctx.in_skipped_block = False
                if str_op_address in ctx.start_node_name_of_skipped_block:
                    assert ctx.in_skipped_block is False, "skipping of overlapping blocks"
                    ctx.in_skipped_block = True
                    ctx.tensor_cache = result
        except:
            # Looks like the __repr__ call made during IDE debug to display tensor contents does not exit properly,
            # but instead throws an exception. This try...except block handles such a situation.
            # Otherwise the context is stuck in the "in_operator == True" state.
            ctx.in_operator = False
            raise

        ctx.in_operator = False
        return result

    wrapped._original_op = operator
    wrapped._operator_namespace = operator_info.operator_namespace
    return wrapped


def wrap_module_call(module_call):
    from nncf.torch.dynamic_graph.patch_pytorch import ORIGINAL_OPERATORS

    NAMES_ORIGINAL_OPERATORS = [op.name for op in ORIGINAL_OPERATORS]

    @functools.wraps(module_call)
    def wrapped(self, *args, **kwargs):
        from nncf.torch.dynamic_graph.patch_pytorch import unpatching_module_call

        # If called on a model compiled by torch dynamo, we unpatch torch operators and invoke original module call
        if "_torchdynamo_orig_callable" in self.forward.__dict__:
            return unpatching_module_call(self, *args, **kwargs)

        ctx = get_current_context()
        if not ctx or self.__class__ in _IGNORED_SCOPES:
            if isinstance(self, DataParallel):
                _warn_data_parallel()
            return module_call(self, *args, **kwargs)
        ctx.push_scope(self)
        is_nncf_layer = isinstance(self, _NNCFModuleMixin)
        if is_nncf_layer:
            op_name = self.op_func_name
        else:
            op_name = self.__class__.__name__
        is_layer_or_op = op_name in NAMES_ORIGINAL_OPERATORS or is_nncf_layer
        op_address = ctx.get_caller_context(op_name)
        str_op_address = str(op_address)
        if ctx.elastic_depth and is_layer_or_op and ctx.in_skipped_block:
            ctx.register_operator_call(op_address.operator_name, op_address.scope_in_model)
            retval = ctx.tensor_cache
            if str_op_address in ctx.end_node_name_of_skipped_block:
                assert ctx.in_skipped_block is True
                ctx.in_skipped_block = False
            if str_op_address in ctx.start_node_name_of_skipped_block:
                assert ctx.in_skipped_block is False, "skipping of overlapping blocks"
                ctx.in_skipped_block = True
        else:
            retval = module_call(self, *args, **kwargs)

        if type(self).__name__ in ITERATION_MODULES.registry_dict:
            ctx.reset_operator_call_count_in_scope(ctx.scope)
        ctx.pop_scope()
        return retval

    return wrapped


def _execute_op(
    op_address: OperationAddress,
    operator_info: PatchedOperatorInfo,
    operator: Callable,
    ctx: TracingContext,
    *args,
    **kwargs,
):
    op_name = operator_info.name

    op_input = OperatorInput(list(args), kwargs)
    op_input = _process_parameters(op_input, ctx)
    processed_input = ctx.execute_pre_hooks(op_address, op_input)
    args = tuple(processed_input.op_args)
    kwargs = processed_input.op_kwargs
    result = operator(*args, **kwargs)
    node = None
    if isinstance(result, type(NotImplemented)):
        nncf_logger.debug("Operation {} returned NotImplemented".format(op_name))
    elif ctx.trace_dynamic_graph:
        tensor_metas = make_tensor_metas(processed_input)
        node = ctx.find_operator_node(tensor_metas, op_address)
        if node is None:
            layer_attrs, ignored_algos = _collect_module_attrs_and_ignored_algorithms(ctx, op_name, args, kwargs)
            is_called_inside_nncf_module = isinstance(ctx.get_current_module(), _NNCFModuleMixin)
            node = ctx.maybe_add_node(
                processed_input, tensor_metas, op_address, layer_attrs, ignored_algos, is_called_inside_nncf_module
            )
        if is_debug() and node is not None:
            ctx.register_node_call(node)

    result = trace_tensors(result, node, ctx)
    result = ctx.execute_post_hooks(op_address, result)
    return result


def _collect_module_attrs_and_ignored_algorithms(
    ctx: TracingContext, op_name: str, args, kwargs
) -> Tuple[BaseLayerAttributes, List[str]]:
    ignored_algos = []
    layer_attrs = get_layer_attributes_from_args_and_kwargs(op_name, args, kwargs)

    curr_module = ctx.get_current_module()
    if curr_module is not None:
        if isinstance(curr_module, _NNCFModuleMixin):
            ignored_algos = deepcopy(curr_module.ignored_algorithms)

        if (
            isinstance(layer_attrs, WeightedLayerAttributes)
            and hasattr(curr_module, "weight_g")
            and hasattr(curr_module, "weight_v")
        ):
            # torch.nn.utils.weight_norm replaces weight with weight_g and weight_v
            layer_attrs.weight_requires_grad = curr_module.weight_g.requires_grad

    return layer_attrs, ignored_algos


@functools.partial(wrap_operator, operator_info=PatchedOperatorInfo(MODEL_CONST_OP_NAME, NamespaceTarget.EXTERNAL))
def process_parameter_fn(x: torch.nn.Parameter) -> torch.nn.Parameter:
    """
    The identity binding function to trace and apply hooks to parameters.

    :param x: A parameter.
    :return: A parameter.
    """
    return x


def _process_parameters(operator_inputs: OperatorInput, ctx: TracingContext) -> OperatorInput:
    """
    Process model parameters into operator inputs applying registered hooks to them. The function guarantees
    that the parameter is processed once.

    :param operator_inputs: The operator inputs.
    :param ctx: The compression context.
    :return: The operator inputs with processed parameters.
    """
    if ctx.in_parameter_trace:
        return operator_inputs

    in_op = getattr(ctx, "in_operator", False)
    ctx.in_operator = False

    for idx in range(len(operator_inputs)):
        traced_parameter = operator_inputs[idx]
        if not isinstance(traced_parameter, TracedParameter):
            continue

        processed_parameter = ctx.get_processed_parameter(traced_parameter.name)
        if processed_parameter is not None:
            operator_inputs[idx] = processed_parameter
            continue

        if traced_parameter.tensor_meta is not None:
            continue

        in_parameter_trace = getattr(ctx, "in_parameter_trace", False)
        ctx.in_parameter_trace = True
        is_reused = traced_parameter.is_reused
        processed_parameter = process_parameter_fn(traced_parameter)
        operator_inputs[idx] = processed_parameter
        if is_reused:
            ctx.register_processed_parameter(traced_parameter.name, processed_parameter)
        ctx.in_parameter_trace = in_parameter_trace

    ctx.in_operator = in_op
    return operator_inputs


def wrap_parameters(model: torch.nn.Module):
    """
    Wrap model parameters inplace by adding tracing capabilities.

    :param model: A model.
    """
    ctx = get_current_context()
    for name, param in model.named_parameters():
        if name.startswith("_nncf"):
            # Exclude parameters in modules which added by NNCF.
            continue
        is_reused = name in ctx.reused_parameters
        tt = TracedParameter.from_torch_parameter(param, name, is_reused)
        ctx.register_traced_tensor(tt)
