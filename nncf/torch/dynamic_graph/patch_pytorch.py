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

from enum import Enum

from typing import List

import warnings

import torch
import torch.utils.cpp_extension
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from nncf.common.utils.logger import logger
from nncf.common.utils.os import safe_open
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
from nncf.torch.dynamic_graph.wrappers import ignore_scope
from nncf.torch.dynamic_graph.wrappers import wrap_module_call
from nncf.torch.dynamic_graph.wrappers import wrap_operator


class NamespaceTarget(Enum):
    """
    NamespaceTarget stores modules from which patched operators were obtained.
    """
    TORCH_NN_FUNCTIONAL = 'torch.nn.functional'
    TORCH_TENSOR = 'torch.tensor'
    TORCH = 'torch'
    EXTERNAL = 'external_function'


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
    TENSOR_CREATING_FUNCTIONS = ['arange', 'as_subclass', 'as_tensor', 'copysign', 'copysign_', 'detach', 'detach_',
                                 'empty', 'ones', 'ones_like', 'rad2deg', 'rad2deg_', 'rand', 'randn', 'randn_like',
                                 'tensor', 'zeros']
    TENSOR_UTILITY_FUNCTIONS = ['all', 'allclose', 'any', 'assert_int_or_pair',
                                'backward', 'broadcast_to', 'cpu', 'cuda', 'data_ptr', 'dequantize', 'dim',
                                'handle_torch_function', 'has_names', 'has_torch_function', 'has_torch_function_unary',
                                'has_torch_function_variadic', 'is_contiguous', 'item', 'names', 'numel', 'numpy',
                                'q_per_channel_axis', 'q_per_channel_scales', 'q_per_channel_zero_points', 'q_scale',
                                'q_zero_point', 'qr', 'qscheme', 'random_', 'record_stream', 'refine_names',
                                'register_hook', 'rename', 'rename_', 'shape', 'size', 'sort', 'storage',
                                'storage_offset', 'stride', 'to']

    FUNCTIONS_TO_PATCH_WITHOUT_TRACING = TENSOR_CREATING_FUNCTIONS + TENSOR_UTILITY_FUNCTIONS


class MagicFunctionsToPatch:
    MAGIC_FUNCTIONS_TO_PATCH = {
        NamespaceTarget.TORCH_TENSOR: ["__add__", "__iadd__", "__radd__", "__sub__", "__isub__",
                                       "__rsub__", "__mul__",
                                       "__imul__", "__rmul__", "__div__", "__idiv__",
                                       "__truediv__", "__floordiv__",
                                       "__ifloordiv__", "__rfloordiv__", "__getitem__",
                                       "__lt__", "__le__", "__gt__",
                                       "__ge__", "__mod__", "__eq__", "__ne__", "__or__",
                                       "__xor__", "__and__", "__pow__"]
    }


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
    # so at each import of a @torch.jit.script-decorated
    # function we need to un-patch the torch operators
    unpatch_torch_operators()

    retval = _ORIG_JIT_SCRIPT(*args, **kwargs)
    patch_torch_operators()
    return retval


class OriginalOpInfo:
    def __init__(self, name: str, namespace, op):
        self.name = name
        self.namespace = namespace
        self.op = op


ORIGINAL_OPERATORS = []  # type: List[OriginalOpInfo]
_JIT_ALREADY_WRAPPED = False
_OPERATORS_ALREADY_WRAPPED = False
_ORIG_JIT_SCRIPT = None


def patch_torch_jit_script():
    # This import statement is required, otherwise we get a
    # "RuntimeError: undefined value torch" inside the real torch.jit.script
    # pylint:disable=unused-import,redefined-outer-name,reimported
    import torch

    orig = getattr(torch.jit, "script")
    global _ORIG_JIT_SCRIPT
    _ORIG_JIT_SCRIPT = orig
    setattr(torch.jit, "script", torch_jit_script_wrapper)


def patch_namespace_opname(namespace, op_info: PatchedOperatorInfo):
    op_name = op_info.name
    if hasattr(namespace, op_name):
        orig = getattr(namespace, op_name)
        ORIGINAL_OPERATORS.append(OriginalOpInfo(op_name, namespace, orig))
        setattr(namespace, op_name, wrap_operator(orig, op_info))
    else:
        warnings.warn("Not patching {} since it is missing in this version of PyTorch".format(op_name))


def get_all_functions_from_namespace(namespace: NamespaceTarget, do_filter: bool = True) -> List[str]:
    """
    Seeks all attributes from the namespace, then takes only attributes,
    which types are function, builtin, method or method descriptor.
    If 'do_filer' is True, then also removes all private or magic attributes.
    :param namespace: Python module.
    :param do_filter: If True return only public functions, else - otherwise.
    """
    import inspect

    def remove_private_functions(names: List[str]) -> List[str]:
        filtered_names = []
        for name in names:
            if name.startswith('_'):
                continue
            filtered_names.append(name)
        return filtered_names

    patched_namespace = get_namespace_to_extract_functions_from(namespace)
    all_torch_function_names = []
    members = inspect.getmembers(patched_namespace)
    for member in members:
        if inspect.isfunction(member[1]) or inspect.isbuiltin(member[1]) or inspect.ismethod(
                member[1]) or inspect.ismethoddescriptor(member[1]):
            all_torch_function_names.append(member[0])
    if do_filter:
        filtered_function_names = remove_private_functions(all_torch_function_names)
        return filtered_function_names
    return all_torch_function_names


def patch_torch_operators():
    # Only patch torch.jit.script during first patch_torch_operators call
    global _JIT_ALREADY_WRAPPED
    if not _JIT_ALREADY_WRAPPED:
        patch_torch_jit_script()
        _JIT_ALREADY_WRAPPED = True

    # Do not patch operators twice as well
    global _OPERATORS_ALREADY_WRAPPED
    if _OPERATORS_ALREADY_WRAPPED:
        return
    _OPERATORS_ALREADY_WRAPPED = True

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

    functions_to_patch[NamespaceTarget.TORCH_NN_FUNCTIONAL].remove('relu')

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


def patch_extension_build_function():
    """
    The function patches PyTorch and fix a bug inside CUDA extensions building;
    The bug must be fixed with a new PyTorch 1.8.0
    """
    try:
        torch_version_numbers = torch.__version__.split('+', maxsplit=1)[0]
        split_torch_version = list(map(int, torch_version_numbers.split('.')))
    except ValueError as e:
        logger.warning('Skip applying a patch to building extension with a reason: '
                       'Cannot parse a PyTorch version with the error {}'.format(e))
        return

    if split_torch_version < [1, 8, 0]:
        if torch.__version__ not in ('1.5.1', '1.7.0', '1.7.1'):
            logger.warning('Skip applying a patch to building extension with a reason: '
                           'PyTorch version is not supported for this')
            return

        def sort_arch_flags(func):
            def wrapped(*args, **kwargs):
                flags = func(*args, **kwargs)
                return sorted(flags)

            return wrapped

        # pylint:disable=protected-access
        torch.utils.cpp_extension._get_cuda_arch_flags = \
            sort_arch_flags(torch.utils.cpp_extension._get_cuda_arch_flags)

    else:
        import re
        import sys
        from pathlib import Path

        # A hackish backport of the https://github.com/pytorch/pytorch/pull/56015 fix.
        def remove_nvcc_dep_build(func):
            def wrapped(*args, **kwargs):
                func(*args, **kwargs)
                if len(args) > 0:
                    target_ninja_file_path = args[0]
                else:
                    target_ninja_file_path = kwargs['path']
                with safe_open(Path(target_ninja_file_path), 'r') as ninja_build_file:
                    ninja_file_contents = ninja_build_file.read()
                with safe_open(Path(target_ninja_file_path), 'w') as ninja_build_file:
                    ninja_build_file.write(re.sub(r'--generate-dependencies-with-compile --dependency-output \$out\.d',
                                                  '', ninja_file_contents))

            return wrapped

        if sys.platform != 'win32':
            # pylint:disable=protected-access
            torch.utils.cpp_extension._write_ninja_file = \
                remove_nvcc_dep_build(torch.utils.cpp_extension._write_ninja_file)
