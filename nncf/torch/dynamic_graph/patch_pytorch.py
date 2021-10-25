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

from typing import List

import warnings

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from nncf.common.utils.logger import logger
from nncf.common.utils.os import safe_open
from nncf.torch.dynamic_graph.trace_functions import CustomTraceFunction
from nncf.torch.dynamic_graph.trace_functions import ForwardTraceOnly
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
from nncf.torch.dynamic_graph.wrappers import ignore_scope
from nncf.torch.dynamic_graph.wrappers import wrap_module_call
from nncf.torch.dynamic_graph.wrappers import wrap_operator


class PatchedOperatorInfo:
    def __init__(self, name: str, custom_trace_fn: CustomTraceFunction = None):
        """
        custom_trace_fn will be called instead of the regular node search/insertion step
        during the corresponding operator call
        """
        self.name = name
        self.custom_trace_fn = custom_trace_fn


def register_operator(name=None):
    def wrap(operator):
        op_name = name
        if op_name is None:
            op_name = operator.__name__
        return wrap_operator(operator, PatchedOperatorInfo(op_name))

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


def patch_namespace_opname(namespace, patched_op_info: PatchedOperatorInfo):
    name = patched_op_info.name
    if hasattr(namespace, name):
        orig = getattr(namespace, name)
        ORIGINAL_OPERATORS.append(OriginalOpInfo(name, namespace, orig))
        setattr(namespace, name, wrap_operator(orig, patched_op_info))
    else:
        warnings.warn("Not patching {} since it is missing in this version of PyTorch"
                      .format(name))


def patch_namespace_by_patchspec(namespace, patchspec: 'PatchSpec'):
    for op_name in patchspec.underlying_function_names:
        patched_op_info = PatchedOperatorInfo(op_name, patchspec.custom_trace_fn)
        patch_namespace_opname(namespace, patched_op_info)


def get_all_functions_from_namespace(namespace: object, do_filter: bool = True) -> List[str]:
    import inspect

    def remove_private_functions(names: List[str]) -> List[str]:
        filtered_names = []
        for name in names:
            if name.startswith('_'):
                continue
            filtered_names.append(name)
        return filtered_names

    all_torch_function_names = []
    members = inspect.getmembers(namespace)
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

    # patch operators
    import torch
    from nncf.torch.graph.operator_metatypes import get_operator_metatypes
    from nncf.torch.graph.operator_metatypes import PTPatchSpec
    # pylint: disable=protected-access
    functions_to_patch = {torch.nn.functional: get_all_functions_from_namespace(torch.nn.functional),
                          torch: get_all_functions_from_namespace(torch._C._VariableFunctions),
                          TracedTensor: get_all_functions_from_namespace(torch.Tensor)}
    # pylint: enable=protected-access
    creating_tensor_funcs = ['arange', 'as_tensor', 'empty', 'rand', 'randn', 'ones', 'tensor', 'zeros', 'ones_like', 'rad2deg',
                             'rad2deg_', 'randn_like', 'as_subclass', 'copy_', 'clone', 'copysign', 'copysign_',
                             'contiguous', 'detach', 'detach_']
    utility_tensor_funcs = ['all', 'allclose', 'any', 'backward', 'broadcast_to', 'dim', 'names', 'rename', 'rename_',
                            'refine_names', 'register_hook', 'record_stream', 'random_', 'cpu', 'cuda', 'data_ptr',
                            'dequantize', 'qscheme', 'q_per_channel_axis', 'q_per_channel_scales',
                            'q_per_channel_zero_points', 'q_scale', 'q_zero_point', 'qr',
                            'view', 'size', 'shape', 'has_names', '_reduce_ex_internal', '__reduce_ex__',
                            'storage', 'sort', 'storage_offset', 'stride', 'item', 'numpy',
                            'is_contiguous', 'has_torch_function_unary', 'has_torch_function_variadic',
                            'assert_int_or_pair', 'handle_torch_function', 'has_torch_function', 'to'
                            ]

    type_tensor_func = ['bfloat16', 'bool', 'byte', 'char', 'double']
    ignored_functions = [func for funcs in [creating_tensor_funcs, utility_tensor_funcs, type_tensor_func] for func in
                         funcs]

    for namespace, function_names in functions_to_patch.items():
        # Do iterate and remove
        new_function_names = []
        for function_name in function_names:
            if function_name in ignored_functions:
                continue
            new_function_names.append(function_name)
        functions_to_patch[namespace] = new_function_names

    # Just to have backward compatibility with previous version of NNCF,
    # where 'relu' wasn't traced in torch.nn.functional
    functions_to_patch[torch.nn.functional].remove('relu')

    for namespace, function_names in functions_to_patch.items():
        for function_name in function_names:
            patch_namespace_by_patchspec(namespace, PTPatchSpec([function_name]))

    for op_meta_class in get_operator_metatypes():  # type: OperatorMetatype
        if op_meta_class.torch_tensor_patch_spec is not None:
            ps = op_meta_class.torch_tensor_patch_spec
            patch_namespace_by_patchspec(TracedTensor, ps)

    # Patch __repr__ methods so that debugging does not add new nodes to the graph
    patch_namespace_opname(TracedTensor, PatchedOperatorInfo("__repr__", ForwardTraceOnly()))

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
    import torch.utils.cpp_extension
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
