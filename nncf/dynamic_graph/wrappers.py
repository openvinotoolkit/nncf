import warnings

from torch.nn import DataParallel, Module as TorchModule

from nncf.common.graph.module_attributes import ConvolutionModuleAttributes
from nncf.common.graph.module_attributes import GroupNormModuleAttributes
from nncf.debug import is_debug
from nncf.dynamic_graph.context import get_current_context, OperatorInput
from nncf.graph.graph import ModuleAttributes
from nncf.dynamic_graph.trace_tensor import make_tensor_metas
from nncf.dynamic_graph.trace_tensor import trace_tensors
from nncf.layers import ITERATION_MODULES
from nncf.layers import NNCF_GENERAL_CONV_MODULES_DICT

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


OP_NAMES_REQUIRING_MODULE_ATTRS = [v.op_func_name for v in NNCF_GENERAL_CONV_MODULES_DICT] + ["group_norm"]


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

        if operator_info.custom_trace_fn is not None:
            try:
                result = operator_info.custom_trace_fn(operator, *args, **kwargs)
            except:
                # Looks like the __repr__ call made during IDE debug to display tensor contents does not exit properly,
                # but instead throws an exception. This try...except block handles such a situation.
                # Otherwise the context is stuck in the "in_operator == True" state.
                ctx.in_operator = False
                raise
        else:
            op_name = operator_info.name
            ia_op_exec_context = ctx.get_caller_context(op_name)

            module_attrs = None
            # Collect module attributes, if required
            if op_name in OP_NAMES_REQUIRING_MODULE_ATTRS:
                curr_module = ctx.get_current_module()
                if curr_module is None:
                    raise RuntimeError("Operation {} requires module attributes, "
                                       "but it was executed outside any module".format(op_name))
                module_attrs = _get_module_attributes(curr_module, op_name)

            ctx.register_operator_call(ia_op_exec_context.operator_name, ia_op_exec_context.scope_in_model)
            op_input = OperatorInput(list(args), kwargs)
            processed_input = ctx.execute_pre_hooks(ia_op_exec_context, op_input)

            tensor_metas = make_tensor_metas(processed_input)
            node = ctx.find_operator_node(tensor_metas, ia_op_exec_context)

            if node is None:
                node = ctx.maybe_add_node(processed_input, tensor_metas, ia_op_exec_context, module_attrs)

            if is_debug():
                ctx.register_node_call(ctx.graph.get_node_key_by_id(node.node_id))

            args = tuple(processed_input.op_args)
            kwargs = processed_input.op_kwargs
            result = operator(*args, **kwargs)

            if node is not None:
                result = trace_tensors(result, node)
            result = ctx.execute_post_hooks(ia_op_exec_context, result)

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


def _get_module_attributes(module: TorchModule, operator_name: str) -> ModuleAttributes:
    if operator_name == "group_norm":
        return GroupNormModuleAttributes(
            module.weight.requires_grad,
            module.num_channels,
            module.num_groups
        )
    return ConvolutionModuleAttributes(
        module.weight.requires_grad,
        module.in_channels,
        module.out_channels,
        module.kernel_size,
        module.stride,
        module.groups
    )
