# Copyright (c) 2023 Intel Corporation
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
import inspect
from typing import List

import torch
import torch.utils.cpp_extension
from torch._jit_internal import createResolutionCallbackFromFrame
from torch.jit import is_tracing
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from nncf import nncf_logger
from nncf.common.utils.api_marker import api
from nncf.torch.dynamic_graph.structs import NamespaceTarget
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
from nncf.torch.dynamic_graph.wrappers import ignore_scope
from nncf.torch.dynamic_graph.wrappers import wrap_module_call
from nncf.torch.dynamic_graph.wrappers import wrap_operator


def get_namespace_to_patch(namespace_target: NamespaceTarget) -> object:
    if namespace_target == NamespaceTarget.TORCH_NN_FUNCTIONAL:
        return torch.nn.functional
    if namespace_target == NamespaceTarget.TORCH_TENSOR:
        return TracedTensor
    if namespace_target == NamespaceTarget.TORCH:
        return torch
    raise RuntimeError("{} namespace wasn't found in {}".format(namespace_target, NamespaceTarget))


def get_namespace_to_extract_functions_from(namespace_target: NamespaceTarget) -> object:
    # pylint: disable=protected-access
    if namespace_target == NamespaceTarget.TORCH_NN_FUNCTIONAL:
        return torch.nn.functional
    if namespace_target == NamespaceTarget.TORCH_TENSOR:
        return torch.Tensor
    if namespace_target == NamespaceTarget.TORCH:
        return torch._C._VariableFunctions
    raise RuntimeError("{} namespace wasn't found in {}".format(namespace_target, NamespaceTarget))
    # pylint: enable=protected-access


class PatchedOperatorInfo:
    def __init__(self, name: str, operator_namespace: NamespaceTarget, skip_trace: bool = False):
        """
        Information about patched operator.
        :param name: Operator name
        :param operator_namespace: Python module, from which operator was gotten.
        :param skip_trace: If it is set to True, the both operator and its internal calls
         to otherwise traced functions do not appear into the model graph.
        """
        self.name = name
        self.operator_namespace = operator_namespace
        self.skip_trace = skip_trace


class FunctionsToPatchWithoutTracing:
    TENSOR_CREATING_FUNCTIONS = [
        "arange",
        "as_subclass",
        "as_tensor",
        "copysign",
        "copysign_",
        "detach",
        "detach_",
        "empty",
        "ones",
        "ones_like",
        "rad2deg",
        "rad2deg_",
        "rand",
        "randn",
        "randn_like",
        "tensor",
        "zeros",
    ]
    TENSOR_UTILITY_FUNCTIONS = [
        "all",
        "allclose",
        "any",
        "assert_int_or_pair",
        "backward",
        "broadcast_to",
        "cpu",
        "cuda",
        "data_ptr",
        "dequantize",
        "dim",
        "handle_torch_function",
        "has_names",
        "has_torch_function",
        "has_torch_function_unary",
        "has_torch_function_variadic",
        "is_contiguous",
        "item",
        "names",
        "numel",
        "numpy",
        "q_per_channel_axis",
        "q_per_channel_scales",
        "q_per_channel_zero_points",
        "q_scale",
        "q_zero_point",
        "qr",
        "qscheme",
        "random_",
        "record_stream",
        "refine_names",
        "register_hook",
        "rename",
        "rename_",
        "shape",
        "size",
        "sort",
        "storage",
        "storage_offset",
        "stride",
        "to",
        "get_device",
    ]

    FUNCTIONS_TO_PATCH_WITHOUT_TRACING = TENSOR_CREATING_FUNCTIONS + TENSOR_UTILITY_FUNCTIONS


class MagicFunctionsToPatch:
    MAGIC_FUNCTIONS_TO_PATCH = {
        NamespaceTarget.TORCH_TENSOR: [
            "__add__",
            "__iadd__",
            "__radd__",
            "__sub__",
            "__isub__",
            "__rsub__",
            "__mul__",
            "__matmul__",
            "__rmatmul__",
            "__imul__",
            "__rmul__",
            "__div__",
            "__idiv__",
            "__truediv__",
            "__floordiv__",
            "__ifloordiv__",
            "__rfloordiv__",
            "__getitem__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
            "__mod__",
            "__eq__",
            "__ne__",
            "__or__",
            "__xor__",
            "__and__",
            "__pow__",
        ]
    }


@api(canonical_alias="nncf.torch.register_operator")
def register_operator(name=None):
    def wrap(operator):
        op_name = name
        if op_name is None:
            op_name = operator.__name__
        return wrap_operator(operator, PatchedOperatorInfo(op_name, NamespaceTarget.EXTERNAL))

    return wrap

    # TODO: Use same wrapper for model.forward() calls


def torch_jit_script_wrapper(*args, **kwargs):
    # Torch JIT cannot work with NNCF-modified operators,
    # so at call of torch.jit.script function we need to
    # un-patch the torch operators

    # If already unpatched, don't perform unpatch/patch
    apply_unpatch = _OPERATORS_ALREADY_WRAPPED
    if apply_unpatch:
        unpatch_torch_operators()

    signature = inspect.signature(_ORIG_JIT_SCRIPT)
    bound_args = signature.bind(*args, **kwargs).arguments
    # Process the case when the object-to-script is a class as in the original jit.script logic
    if inspect.isclass(bound_args["obj"]):
        # Inserting wrapper alters the call stack, hence we need to change the resolution callback accordingly
        if "_rcb" not in bound_args:
            frames_up = bound_args.get("_frames_up", 0)
            rcb = createResolutionCallbackFromFrame(frames_up + 1)
            kwargs["_rcb"] = rcb
        retval = _ORIG_JIT_SCRIPT(*args, **kwargs)
    else:
        # For some reason resolution callback may return patched methods, so we wrap it to avoid this
        if "_rcb" in kwargs:
            rcb = kwargs["_rcb"]

            def rcb_wrapper(name):
                value = rcb(name)
                if hasattr(value, "_original_op"):
                    value = value._original_op  # pylint: disable=protected-access
                return value

            kwargs["_rcb"] = rcb_wrapper

        retval = _ORIG_JIT_SCRIPT(*args, **kwargs)

    if apply_unpatch:
        patch_torch_operators()

    return retval


def torch_jit_trace_make_module_wrapper(*args, **kwargs):
    apply_unpatch = _OPERATORS_ALREADY_WRAPPED
    if apply_unpatch:
        unpatch_torch_operators()
    retval = _ORIG_JIT_TRACE_MAKE_MODULE(*args, **kwargs)
    if apply_unpatch:
        patch_torch_operators()
    return retval


def torch_jit_script_if_tracing(fn):
    # pylint: disable=protected-access
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_tracing():
            return fn(*args, **kwargs)

        compiled_fn = torch.jit.script(wrapper.__original_fn)
        return compiled_fn(*args, **kwargs)

    wrapper.__original_fn = fn
    wrapper.__script_if_tracing_wrapper = True

    return wrapper


class OriginalOpInfo:
    def __init__(self, name: str, namespace, op):
        self.name = name
        self.namespace = namespace
        self.op = op


ORIGINAL_OPERATORS = []  # type: List[OriginalOpInfo]
ORIGINAL_CALL = torch.nn.Module.__call__
_JIT_ALREADY_WRAPPED = False
_OPERATORS_ALREADY_WRAPPED = False
_ORIG_JIT_SCRIPT = None
_ORIG_JIT_TRACE_MAKE_MODULE = None


def patch_torch_jit():
    # This import statement is required, otherwise we get a
    # "RuntimeError: undefined value torch" inside the real torch.jit.script
    # pylint:disable=unused-import,redefined-outer-name,reimported,protected-access
    import torch

    global _ORIG_JIT_SCRIPT
    _ORIG_JIT_SCRIPT = getattr(torch.jit, "script")
    setattr(torch.jit, "script", torch_jit_script_wrapper)

    # Patch torch.jit._trace.make_module() which is called during
    # torch.jit.trace() call
    global _ORIG_JIT_TRACE_MAKE_MODULE
    _ORIG_JIT_TRACE_MAKE_MODULE = getattr(torch.jit._trace, "make_module")
    setattr(torch.jit._trace, "make_module", torch_jit_trace_make_module_wrapper)

    # Patch torch.jit._script_if_tracing because it references an original
    # unpatched torch.jit.script and the patching above does not affect it
    setattr(torch.jit, "_script_if_tracing", torch_jit_script_if_tracing)


def patch_namespace_opname(namespace, op_info: PatchedOperatorInfo):
    op_name = op_info.name
    if hasattr(namespace, op_name):
        orig = getattr(namespace, op_name)
        ORIGINAL_OPERATORS.append(OriginalOpInfo(op_name, namespace, orig))
        setattr(namespace, op_name, wrap_operator(orig, op_info))
    else:
        nncf_logger.debug(f"Not patching {op_name} since it is missing in this version of PyTorch")


def get_all_functions_from_namespace(namespace: NamespaceTarget, do_filter: bool = True) -> List[str]:
    """
    Seeks all attributes from the namespace, then takes only attributes,
    which types are function, builtin, method or method descriptor.
    If 'do_filer' is True, then also removes all private or magic attributes.
    :param namespace: Python module.
    :param do_filter: If True return only public functions, else - otherwise.
    """

    def remove_private_functions(names: List[str]) -> List[str]:
        filtered_names = []
        for name in names:
            if name.startswith("_"):
                continue
            filtered_names.append(name)
        return filtered_names

    patched_namespace = get_namespace_to_extract_functions_from(namespace)
    all_torch_function_names = []
    members = inspect.getmembers(patched_namespace)
    for member in members:
        if (
            inspect.isfunction(member[1])
            or inspect.isbuiltin(member[1])
            or inspect.ismethod(member[1])
            or inspect.ismethoddescriptor(member[1])
        ):
            all_torch_function_names.append(member[0])
    if do_filter:
        filtered_function_names = remove_private_functions(all_torch_function_names)
        return filtered_function_names
    return all_torch_function_names


def patch_torch_operators():
    # Only patch torch.jit.script during first patch_torch_operators call
    global _JIT_ALREADY_WRAPPED
    if not _JIT_ALREADY_WRAPPED:
        patch_torch_jit()
        _JIT_ALREADY_WRAPPED = True

    # Do not patch operators twice as well
    global _OPERATORS_ALREADY_WRAPPED
    if _OPERATORS_ALREADY_WRAPPED:
        return
    _OPERATORS_ALREADY_WRAPPED = True

    global ORIGINAL_OPERATORS
    ORIGINAL_OPERATORS = []

    functions_to_patch = {}
    for namespace in NamespaceTarget:
        if namespace == NamespaceTarget.EXTERNAL:
            continue
        functions_to_patch[namespace] = get_all_functions_from_namespace(namespace)

    ignored_functions = FunctionsToPatchWithoutTracing.FUNCTIONS_TO_PATCH_WITHOUT_TRACING
    functions_to_patch_without_tracing = {}
    for namespace, function_names in functions_to_patch.items():
        functions_to_patch_without_tracing[namespace] = []
        new_function_names = []
        for function_name in function_names:
            if function_name in ignored_functions:
                functions_to_patch_without_tracing[namespace].append(function_name)
                continue
            new_function_names.append(function_name)
        functions_to_patch[namespace] = new_function_names

    # Relu function from torch.nn.functional uses torch.relu or torch.relu_ inside,
    # which we've already wrapped. So we don't have to wrap Relu in torch.nn.functional
    # to be able to differ what torch.relu or torch.relu was called exactly inside Relu
    functions_to_patch[NamespaceTarget.TORCH_NN_FUNCTIONAL].remove("relu")

    # nn.MultiheadAttention uses multi_head_attention_forward from torch.nn.functional inside, which
    # leads to adding a single node to the graph. In turn, it makes impossible to quantize internals of the multi head
    # attention which is required for better speed-up. So multi_head_attention_forward is skipped from wrapping and
    # tracing goes inside and finds internal operations in it: bmm, linear, softmax and etc.
    functions_to_patch[NamespaceTarget.TORCH_NN_FUNCTIONAL].remove("multi_head_attention_forward")

    magic_functions_to_patch = MagicFunctionsToPatch.MAGIC_FUNCTIONS_TO_PATCH
    for namespace, function_names in magic_functions_to_patch.items():
        functions_to_patch[namespace] += function_names

    for namespace, function_names in functions_to_patch.items():
        for function_name in function_names:
            op_info = PatchedOperatorInfo(function_name, namespace)
            patched_namespace = get_namespace_to_patch(namespace)
            patch_namespace_opname(patched_namespace, op_info)

    # Patch operators without tracing so that
    # both they and any internal calls to otherwise traced functions do not appear into the model graph.

    for namespace, function_names in functions_to_patch_without_tracing.items():
        for function_name in function_names:
            op_info = PatchedOperatorInfo(function_name, namespace, skip_trace=True)
            patched_namespace = get_namespace_to_patch(namespace)
            patch_namespace_opname(patched_namespace, op_info)

    # Patch __repr__ twice in 'torch.Tensor' and 'TracedTensor'.
    # This is done to not add operations behind print() operator for the both TracedTensor and torch.Tensor.

    op_info = PatchedOperatorInfo("__repr__", NamespaceTarget.TORCH_TENSOR, skip_trace=True)
    patch_namespace_opname(torch.Tensor, op_info)
    op_info = PatchedOperatorInfo("__repr__", NamespaceTarget.TORCH_TENSOR, skip_trace=True)
    patch_namespace_opname(TracedTensor, op_info)

    ORIGINAL_OPERATORS.append(OriginalOpInfo("__call__", torch.nn.Module, torch.nn.Module.__call__))
    torch.nn.Module.__call__ = wrap_module_call(torch.nn.Module.__call__)
    ignore_scope(DataParallel)
    ignore_scope(DistributedDataParallel)


def unpatch_torch_operators():
    global _OPERATORS_ALREADY_WRAPPED
    if not _OPERATORS_ALREADY_WRAPPED:
        return
    _OPERATORS_ALREADY_WRAPPED = False

    for orig_op_info in ORIGINAL_OPERATORS:
        setattr(orig_op_info.namespace, orig_op_info.name, orig_op_info.op)
