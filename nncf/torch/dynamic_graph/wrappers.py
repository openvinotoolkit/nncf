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

import warnings
from copy import deepcopy

from torch.nn import Conv1d
from torch.nn import Conv2d
from torch.nn import Conv3d
from torch.nn import ConvTranspose1d
from torch.nn import ConvTranspose2d
from torch.nn import ConvTranspose3d
from torch.nn import DataParallel, Module as TorchModule
from torch.nn import Linear

from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.debug import is_debug
from nncf.torch.dynamic_graph.context import get_current_context
from nncf.torch.dynamic_graph.op_input_processing import OperatorInput
from nncf.torch.dynamic_graph.trace_tensor import make_tensor_metas
from nncf.torch.dynamic_graph.trace_tensor import trace_tensors
from nncf.torch.layer_utils import _NNCFModuleMixin
from nncf.torch.layers import ITERATION_MODULES
from nncf.torch.layers import NNCF_MODULES_DICT

_IGNORED_SCOPES = []


def _warn_data_parallel():
    if getattr(_warn_data_parallel, 'warned_once', False):
        return
    _warn_data_parallel.warned_once = True
    warnings.warn("You are using DataParallel, which may cause significant performance issues with dynamic graph "
                  "building. Consider using distributed training (DistributedDataParallel) instead")


def ignore_scope(cls):
    if cls not in _IGNORED_SCOPES:
        _IGNORED_SCOPES.append(cls)
    return cls


OP_NAMES_REQUIRING_MODULE_ATTRS = [v.op_func_name for v in NNCF_MODULES_DICT] + ["group_norm"]


def wrap_operator(operator, operator_info: 'PatchedOperatorInfo'):
    # do not wrap function twice
    _orig_op = getattr(operator, '_original_op', None)
    if _orig_op is not None:
        raise Exception("Operator: {} is already wrapped".format(_orig_op.__name__))

    def wrapped(*args, **kwargs):
        ctx = get_current_context()
        if not ctx or getattr(ctx, 'in_operator', False) or not ctx.is_tracing:
            op1 = operator(*args, **kwargs)
            return op1

        ctx.in_operator = True

        try:
            if operator_info.custom_trace_fn is not None:
                result = operator_info.custom_trace_fn(operator, *args, **kwargs)
            elif ctx.is_forwarding:
                from nncf.torch.dynamic_graph.patch_pytorch import ForwardTraceOnly
                result = ForwardTraceOnly()(operator, *args, **kwargs)
            else:
                op_name = operator_info.name
                op_address = ctx.get_caller_context(op_name)

                layer_attrs = None
                ignored_algos = []
                # Collect module attributes, if required
                if op_name in OP_NAMES_REQUIRING_MODULE_ATTRS:
                    curr_module = ctx.get_current_module()
                    if curr_module is None:
                        raise RuntimeError("Operation {} requires module attributes, "
                                           "but it was executed outside any module".format(op_name))
                    layer_attrs = _get_layer_attributes(curr_module, op_name)
                    if isinstance(curr_module, _NNCFModuleMixin):
                        ignored_algos = deepcopy(curr_module.ignored_algorithms)

                ctx.register_operator_call(op_address.operator_name, op_address.scope_in_model)
                op_input = OperatorInput(list(args), kwargs)
                processed_input = ctx.execute_pre_hooks(op_address, op_input)

                tensor_metas = make_tensor_metas(processed_input)
                node = ctx.find_operator_node(tensor_metas, op_address)

                args = tuple(processed_input.op_args)
                kwargs = processed_input.op_kwargs
                result = operator(*args, **kwargs)

                if isinstance(result, type(NotImplemented)):
                    nncf_logger.debug("Operation {} returned NotImplemented".format(op_name))
                elif node is None:
                    node = ctx.maybe_add_node(processed_input, tensor_metas, op_address, layer_attrs, ignored_algos)

                if node is not None:
                    if is_debug():
                        ctx.register_node_call(node)
                    result = trace_tensors(result, node)
                result = ctx.execute_post_hooks(op_address, result)
        except:
            # Looks like the __repr__ call made during IDE debug to display tensor contents does not exit properly,
            # but instead throws an exception. This try...except block handles such a situation.
            # Otherwise the context is stuck in the "in_operator == True" state.
            ctx.in_operator = False
            raise

        ctx.in_operator = False
        return result

    # pylint: disable=protected-access
    wrapped._original_op = operator
    return wrapped


def wrap_module_call(module_call):
    def wrapped(self, *args, **kwargs):
        ctx = get_current_context()
        if not ctx or self.__class__ in _IGNORED_SCOPES:
            if isinstance(self, DataParallel):
                _warn_data_parallel()
            return module_call(self, *args, **kwargs)
        ctx.push_scope(self)
        retval = module_call(self, *args, **kwargs)
        if type(self).__name__ in ITERATION_MODULES.registry_dict.keys():
            ctx.reset_operator_call_count_in_scope(ctx.scope)
        ctx.pop_scope()
        return retval

    return wrapped


def _get_layer_attributes(module: TorchModule, operator_name: str) -> BaseLayerAttributes:
    if operator_name == "group_norm":
        return GroupNormLayerAttributes(
            module.weight.requires_grad,
            module.num_channels,
            module.num_groups
        )
    if isinstance(module, (Conv1d, Conv2d, Conv3d)):
        return ConvolutionLayerAttributes(weight_requires_grad=module.weight.requires_grad,
                                          in_channels=module.in_channels,
                                          out_channels=module.out_channels,
                                          kernel_size=module.kernel_size,
                                          stride=module.stride,
                                          groups=module.groups,
                                          transpose=False,
                                          padding_values=module.padding)
    if isinstance(module, (ConvTranspose1d, ConvTranspose2d, ConvTranspose3d)):
        return ConvolutionLayerAttributes(weight_requires_grad=module.weight.requires_grad,
                                          in_channels=module.in_channels,
                                          out_channels=module.out_channels,
                                          kernel_size=module.kernel_size,
                                          stride=module.stride,
                                          groups=module.groups,
                                          transpose=True,
                                          padding_values=module.padding)
    if isinstance(module, Linear):
        return LinearLayerAttributes(weight_requires_grad=module.weight.requires_grad,
                                     in_features=module.in_features,
                                     out_features=module.out_features)

    if hasattr(module, 'weight'):
        return GenericWeightedLayerAttributes(weight_requires_grad=module.weight.requires_grad,
                                              weight_shape=module.weight.shape)

    return GenericWeightedLayerAttributes(weight_requires_grad=False,
                                          weight_shape=[1, 1])
