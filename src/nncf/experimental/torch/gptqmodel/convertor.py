# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from typing import TypeVar, Union

import torch
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear  # type: ignore[import-not-found]
from torch import nn

import nncf
from nncf.common.logging.track_progress import track
from nncf.torch.function_hook.hook_storage import decode_hook_name
from nncf.torch.function_hook.wrapper import ATR_HOOK_STORAGE
from nncf.torch.function_hook.wrapper import ForwardWithHooks
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.quantization.layers import INT8AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8SymmetricWeightsDecompressor

T_INT8_DECOMPRESSOR = Union[INT8AsymmetricWeightsDecompressor, INT8SymmetricWeightsDecompressor]

TModel = TypeVar("TModel", bound=nn.Module)


def convert_linear_module(linear: nn.Linear, decompressor: T_INT8_DECOMPRESSOR) -> TritonV2QuantLinear:
    """
    Replaces the linear layer weights with compressed INT8 weights to GPTQ format.

    :param linear: The linear layer with compressed weight.
    :param decompressor: The decompressor module.
    :return: The converted GPTQModel linear layer with compressed weights.
    """
    num_bits = 8
    group_size = -1
    if isinstance(decompressor, INT8AsymmetricWeightsDecompressor):
        is_symmetric = False
        sc = decompressor._scale.to(torch.float16).cpu().contiguous()  # type: ignore[operator]
        zp = decompressor._zero_point.to(torch.float16).cpu().contiguous()  # type: ignore[operator]
    elif isinstance(decompressor, INT8SymmetricWeightsDecompressor):
        is_symmetric = True
        sc = decompressor._scale.to(torch.float16).cpu().contiguous()  # type: ignore[operator]
        zp = torch.zeros_like(sc)
    else:
        msg = f"Unsupported decompressor module type: {type(decompressor)}"
        raise nncf.InternalError(msg)

    unpacked_w = decompressor(linear.weight).to(torch.float32)
    g_idx = torch.zeros((unpacked_w.shape[1],), device="cpu", dtype=torch.int32)

    module_device = linear.weight.device

    tmp_linear = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
    tmp_linear.weight.data = unpacked_w
    if linear.bias is not None:
        tmp_linear.bias = deepcopy(linear.bias)

    tmp_linear.cpu()

    triton = TritonV2QuantLinear(
        num_bits,
        group_size=group_size,
        desc_act=False,
        sym=is_symmetric,
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
    )

    # Pack weight support only cpu tensor
    triton.pack(tmp_linear, sc, zp, g_idx)
    triton.to(module_device)
    triton.post_init()

    return triton


def convert_model(model: TModel) -> TModel:
    """
    Converts all linear layers in the model with compressed INT8 weights to GPTQ format.

    This function replaces compressed linear layers with `TritonV2QuantLinear` layers,
    removing any NNCF-specific functionality in the process.

    :param model: The model with compressed weights.
    :return: The model with all eligible linear layers converted to GPTQ format and
        without any NNCF-specific functions or wrappers.
    """
    if not isinstance(model.forward, ForwardWithHooks):
        msg = "Expected the model returned by nncf.compress_weights()"
        raise nncf.InternalError(msg)

    hook_storage = get_hook_storage(model)

    for hook_name, hook in track(list(hook_storage.named_hooks()), description="Converting to GPTQ"):
        if not isinstance(hook, (INT8AsymmetricWeightsDecompressor, INT8SymmetricWeightsDecompressor)):
            msg = f"Unsupported decompressor module type {type(hook)}, {hook_name=}"
            raise nncf.InternalError(msg)

        _, op_name, _ = decode_hook_name(hook_name)
        module_name, _ = split_const_name(op_name)
        module = get_module_by_name(module_name, model)

        if not isinstance(module, torch.nn.Linear):
            msg = f"Unsupported module type {type(module).__name__}, {hook_name=}"
            raise nncf.InternalError(msg)

        new_linear = convert_linear_module(module, hook)
        del hook

        if module_name.count(".") == 0:
            # Top-level module
            setattr(model, module_name, new_linear)
        else:
            parent_module_name, module_child_name = module_name.rsplit(".", 1)
            parent_module = get_module_by_name(parent_module_name, model)
            setattr(parent_module, module_child_name, new_linear)

    # Unwrap the model to avoid conflicts with TorchFunctionMode
    model.forward = model.forward.orig_forward
    delattr(model, ATR_HOOK_STORAGE)

    return model
