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
import random
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Generator, List

import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.nn import Module

import nncf
from nncf.common.compression import BaseCompressionAlgorithmController as BaseController
from nncf.common.deprecation import warning_deprecated
from nncf.common.graph import NNCFNodeName
from nncf.common.logging import nncf_logger
from nncf.common.scopes import matches_any
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.dynamic_graph.scope import ScopeElement
from nncf.torch.dynamic_graph.trace_tensor import TracedTensorMixin
from nncf.torch.layer_utils import _NNCFModuleMixin
from nncf.torch.structures import ExecutionParameters


def get_node_name(module, module_name, prefix):
    return "{prefix}/{cls}[{name}]".format(prefix=prefix, cls=module.__class__.__name__, name=module_name)


def get_all_modules(model, prefix=None):
    found = OrderedDict()
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)
        found[full_node_name] = module
        sub_found = get_all_modules(module, prefix=full_node_name)
        if sub_found:
            found.update(sub_found)
    return found


def get_all_modules_by_type(
    model, module_types=None, current_scope=None, ignored_scopes=None, target_scopes=None, memo=None
) -> Dict[Scope, Module]:
    if memo is None:
        memo = set()
    if isinstance(module_types, str):
        module_types = [module_types]
    found = OrderedDict()

    if current_scope is None:
        current_scope = Scope()
        current_scope.push(ScopeElement(model.__class__.__name__))
    for name, module in model.named_children():
        if id(module) in memo:
            continue
        memo.add(id(module))
        child_scope_element = ScopeElement(module.__class__.__name__, name)
        child_scope = current_scope.copy()
        child_scope.push(child_scope_element)

        if matches_any(str(child_scope), ignored_scopes):
            continue

        if target_scopes is None or matches_any(str(child_scope), target_scopes):
            if module_types is None or module_types.count(str(type(module).__name__)) != 0:
                found[child_scope] = module
            sub_found = get_all_modules_by_type(
                module,
                module_types,
                current_scope=child_scope,
                ignored_scopes=ignored_scopes,
                target_scopes=target_scopes,
                memo=memo,
            )
            if sub_found:
                found.update(sub_found)
    return found


def get_state_dict_names_with_modules(
    model: torch.nn.Module, str_types: List[str] = None, prefix=""
) -> Dict[str, torch.nn.Module]:
    found = OrderedDict()
    for name, module in model.named_children():
        full_node_name = "{}{}".format(prefix, name)
        if str_types is not None and type(module).__name__ in str_types:
            found[full_node_name] = module
        sub_found = get_state_dict_names_with_modules(module, str_types, prefix=full_node_name + ".")
        if sub_found:
            found.update(sub_found)
    return found


def get_filters_num(module):
    if isinstance(module, _NNCFModuleMixin):
        return module.weight.size(module.target_weight_dim_for_compression)
    return module.weight.size(0)


def manual_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def is_tracing_state():
    return torch._C._get_tracing_state() is not None


class no_jit_trace:
    def __enter__(self):
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


def fp32_accum_wrapper(func):
    def wrapper(tensor_to_sum, ret_tensor):
        half = tensor_to_sum.dtype == np.float16
        if half:
            tensor_to_sum = tensor_to_sum.astype(np.float32)
        retval = func(tensor_to_sum, ret_tensor)
        if half:
            retval = retval.astype(np.float16)
        return retval

    return wrapper


@fp32_accum_wrapper
def sum_like(tensor_to_sum, ref_tensor):
    """Warning: may modify tensor_to_sum"""
    if ref_tensor.size == 1:
        return tensor_to_sum.sum()

    for dim, size in enumerate(ref_tensor.shape):
        if size == 1:
            if isinstance(tensor_to_sum, np.ndarray):
                tensor_to_sum = tensor_to_sum.sum(dim, keepdims=True)
            else:
                tensor_to_sum = tensor_to_sum.sum(dim, keepdim=True)
    return tensor_to_sum


def get_flat_tensor_contents_string(input_tensor):
    retval = "["
    for idx, el in enumerate(input_tensor.view(-1)):
        if idx >= 10:
            retval += "... (first 10/{} elements shown only) ".format(len(input_tensor.view(-1)))
            break
        retval += "{:.4f}, ".format(el.item())
    retval += "]"
    return retval


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def safe_thread_call(main_call_fn, after_barrier_call_fn=None):
    result = None
    if is_dist_avail_and_initialized():
        if is_main_process():
            result = main_call_fn()
        dist.barrier()
        if not is_main_process():
            result = after_barrier_call_fn() if after_barrier_call_fn else main_call_fn()
    else:
        result = main_call_fn()
    return result


def is_tensor(obj):
    return isinstance(obj, torch.Tensor)


def is_traced_tensor(obj):
    return isinstance(obj, TracedTensorMixin)


class _ModuleState:
    def __init__(self, base_module: Module = None):
        self._training_state: Dict[str, bool] = {}
        self._requires_grad_state: Dict[str, bool] = {}
        if base_module is not None:
            for module_name, module in base_module.named_modules():
                self.training_state[module_name] = module.training

            for param_name, param in base_module.named_parameters():
                self.requires_grad_state[param_name] = param.requires_grad

    @property
    def training_state(self) -> Dict[str, bool]:
        return self._training_state

    @property
    def requires_grad_state(self) -> Dict[str, bool]:
        return self._requires_grad_state


def save_module_state(module: Module) -> _ModuleState:
    return _ModuleState(module)


def load_module_state(base_module: Module, state: _ModuleState, strict=False) -> None:
    for name, module in base_module.named_modules():
        try:
            module.train(state.training_state[name])
        except KeyError as e:
            # KeyError could happen if the modules name were changed during forward
            # (e.g. LSTM block in NNCF examples)
            msg = f"Could not find a module to restore state: {name}"
            nncf_logger.debug(msg)
            if strict:
                raise nncf.InternalError(msg) from e

    for name, param in base_module.named_parameters():
        param.requires_grad = state.requires_grad_state[name]


@contextmanager
def training_mode_switcher(model: Module, is_training: bool = True):
    saved_state = save_module_state(model)
    model.train(is_training)
    try:
        yield
    finally:
        load_module_state(model, saved_state)


def compute_FLOPs_hook(module, input_, output, dict_to_save, module_node_name: NNCFNodeName):
    # WARNING: numpy should be explicitly given np.int64 as dtype, since default integer type on Win is np.int32
    if isinstance(
        module, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d)
    ):
        ks = module.weight.data.shape
        mac_count = np.prod(ks, dtype=np.int64) * np.prod(output.shape[2:], dtype=np.int64)
    elif isinstance(module, nn.Linear):
        if len(input_[0].shape) == 1:
            # In some test cases input tensor could have dimension [N]
            mac_count = input_[0].shape[0] * output.shape[-1]
        else:
            mac_count = np.prod(input_[0].shape[1:], dtype=np.int64) * output.shape[-1]
    else:
        return
    dict_to_save[module_node_name] = 2 * mac_count


def add_domain(name_operator: str) -> str:
    from nncf.torch.compression_method_api import DOMAIN_CUSTOM_OPS_NAME

    return DOMAIN_CUSTOM_OPS_NAME + "::" + name_operator


def default_distributed_wrapper(model: nn.Module, execution_parameters: ExecutionParameters):
    """
    Wrapping model for distributed training with DataParallel or DistributedDataParallel depending on execution mode
    chosen by user.
    :param execution_parameters: structure with necessary execution parameters
    :param model: model to wrap  in accordance with execution mode chosen by user
    :return: wrapped model
    """
    if not execution_parameters or execution_parameters.cpu_only:
        # If execution params is not set or in cpu_only mode model can't be optimized by parallelization
        return model

    current_gpu = execution_parameters.current_gpu
    if not is_dist_avail_and_initialized():
        if current_gpu is not None:
            # ExecutionMode.SINGLE_GPU
            torch.cuda.set_device(current_gpu)
        else:
            # ExecutionMode.GPU_DATAPARALLEL
            model = torch.nn.DataParallel(model)
    else:
        if current_gpu is None:
            # ExecutionMode.DISTRIBUTED
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            # ExecutionMode.MULTIPROCESSING_DISTRIBUTED
            torch.cuda.set_device(current_gpu)
            model = torch.nn.parallel.distributed.DistributedDataParallel(model, device_ids=[current_gpu])
    return model


def default_distributed_unwrapper(model: nn.Module):
    """
    Unwrapping model prepared from distributed training in Pytorch
    (and wrapped with Dataparallel or DistributedDataParallel).
    :param model: model to unwrap.
    :return: model without parallelization
    """
    if isinstance(model, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def rename_legacy_names_in_state_dict(
    state_dict_to_load: Dict[str, Any], legacy_names: List[str], legacy_name: str, new_name: str
):
    for name in legacy_names:
        tensor = state_dict_to_load.pop(name)
        new_key = name.replace(legacy_name, new_name) if new_name not in name else name
        state_dict_to_load[new_key] = tensor

    if legacy_names:
        warning_deprecated(
            "Legacy Batch Norm layer names was detected in checkpoint model state dict."
            " All occurrences of `{}` in nodes names was replaced by `{}`".format(legacy_name, new_name)
        )


LEGACY_VS_NEW_BN_MAP = {
    "BatchNorm1d": "NNCFBatchNorm1d",
    "BatchNorm2d": "NNCFBatchNorm2d",
    "BatchNorm3d": "NNCFBatchNorm3d",
    "NNCFBatchNorm": "NNCFBatchNorm2d",
    "ConvBNActivation": "Conv2dNormActivation",
}


def maybe_convert_legacy_names_in_model_state(state_dict_to_load: Dict[str, Any]) -> None:
    """
    Convert legacy layer names in compressed model state dict in case such names exist.

    :param state_dict_to_load: State dict to convert.
    """
    legacy_names = LEGACY_VS_NEW_BN_MAP.keys()
    matched_legacy_names = {name: [] for name in legacy_names}
    for name_in_state_dict in state_dict_to_load:
        matched = filter(lambda x: x in name_in_state_dict, legacy_names)
        for legacy_name in matched:
            matched_legacy_names[legacy_name].append(name_in_state_dict)
    for old_name, new_name in LEGACY_VS_NEW_BN_MAP.items():
        rename_legacy_names_in_state_dict(state_dict_to_load, matched_legacy_names[old_name], old_name, new_name)


def maybe_convert_legacy_names_in_compress_state(compression_state: Dict[str, Any]) -> None:
    """
    Convert legacy layer names in compression state in case such names exist.

    :param compression_state: Compression state to convert.
    """
    if not compression_state or BaseController.BUILDER_STATE not in compression_state:
        return

    controller_state = compression_state[BaseController.BUILDER_STATE]
    if not controller_state or "quantization" not in controller_state:
        return

    from nncf.torch.quantization.algo import QUANTIZER_BUILDER_STATE_VERSION_SAVE_NAME

    if not controller_state["quantization"].get(QUANTIZER_BUILDER_STATE_VERSION_SAVE_NAME):
        qips = controller_state["quantization"]["quantizer_setup"]["quantization_points"]

        detected_legacy_names = {
            "BatchNorm1d": False,
            "BatchNorm2d": False,
            "BatchNorm3d": False,
            "NNCFBatchNorm": False,
        }

        for point in qips.values():
            name = point["qip"]["target_node_name"]
            for old_name, new_name in LEGACY_VS_NEW_BN_MAP.items():
                if old_name in name and new_name not in name:
                    detected_legacy_names[old_name] = True
                    point["qip"]["target_node_name"] = name.replace(old_name, new_name)
                    break

        for old_name, was_detected in detected_legacy_names.items():
            if was_detected:
                new_name = LEGACY_VS_NEW_BN_MAP[old_name]
                warning_deprecated(
                    "Legacy Batch Norm layer names was detected in quantization setup target"
                    " point names. All occurrences of `{}` in nodes names was replaced by"
                    " `{}`".format(old_name, new_name)
                )


def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get the device on which the first model parameters reside.

    :param model: The PyTorch model.
    :return: The device where the first model parameter reside.
        Default cpu if the model has no parameters.
    """

    try:
        device = next(model.parameters()).device
    except StopIteration:
        # The model had no parameters at all, doesn't matter which device to choose
        device = torch.device("cpu")
    return device


def get_all_model_devices_generator(model: torch.nn.Module) -> Generator[torch.device, None, None]:
    for p in model.parameters():
        yield p.device


def is_multidevice(model: torch.nn.Module) -> bool:
    """
    Checks if the model's parameters are distributed across multiple devices.

    :param model: The PyTorch model.
    :return: True if the parameters reside on multiple devices, False otherwise.
        Default False if the models has no parameters
    """

    device_generator = get_all_model_devices_generator(model)
    try:
        curr_device = next(device_generator)
    except StopIteration:  # no parameters
        return False

    for d in device_generator:
        if d != curr_device:
            return True
    return False


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """
    Get the datatype of the first model parameter.

    :param model: The PyTorch model.
    :return: The datatype of the first model parameter.
        Default to torch.float32 if the model has no parameters.
    """

    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        # The model had no parameters at all, assume FP32
        dtype = torch.float32
    return dtype
