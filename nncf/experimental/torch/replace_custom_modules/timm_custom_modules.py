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

from copy import deepcopy
from typing import Optional

from timm.layers import Linear
from timm.layers.norm_act import BatchNormAct2d
from timm.layers.norm_act import GroupNormAct
from timm.layers.norm_act import LayerNormAct
from torch import nn

from nncf.torch.nncf_module_replacement import replace_modules_by_nncf_modules


def _copy_parameters(src_module: nn.Module, trg_module: nn.Module):
    """
    Copies parameters of a source module to a target module.
    :param src_module: The source module to copy parameters from.
    :param trg_module: The target module to copy parameters to.
    """
    for name, param in src_module.named_parameters():
        setattr(trg_module, name, deepcopy(param))


def _convert_linear(module: Linear) -> nn.Linear:
    """
    Convert Linear module to torch.nn.Linear.

    param module: The module to convert.
    :return nn.Linear: Converted module.
    """
    with_bias = module.bias is not None
    new_ln = nn.Linear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=with_bias,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    _copy_parameters(module, new_ln)
    return new_ln


def _convert_batch_norm_act_2d(module: BatchNormAct2d) -> nn.Sequential:
    """
    Converts a BatchNormAct2d module to an nn.Sequential module that contains nn.BatchNorm2d,
    followed by dropout and activation functions.

    :param module: The module to convert.
    :return nn.Sequential: A new nn.Sequential module containing nn.BatchNorm2d, dropout, and activation functions.
    """
    new_bn = nn.BatchNorm2d(
        num_features=module.num_features,
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    _copy_parameters(module, new_bn)
    new_bn.running_mean = deepcopy(module.running_mean)
    new_bn.running_var = deepcopy(module.running_var)

    new_drop = deepcopy(module.drop)
    new_act = deepcopy(module.act)
    return nn.Sequential(new_bn, new_drop, new_act)


def _convert_group_norm_act(module: GroupNormAct) -> nn.Sequential:
    """
    Converts a GroupNormAct module to an nn.Sequential module that contains nn.GroupNorm,
    followed by dropout and activation functions.

    :param module: The module to convert.
    :return nn.Sequential: A new nn.Sequential module containing nn.GroupNorm, dropout, and activation functions.
    """
    new_gn = nn.GroupNorm(
        num_groups=module.num_groups,
        num_channels=module.num_channels,
        eps=module.eps,
        affine=module.eps,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    _copy_parameters(module, new_gn)
    new_drop = deepcopy(module.drop)
    new_act = deepcopy(module.act)
    return nn.Sequential(new_gn, new_drop, new_act)


def _convert_layer_norm_act(module: LayerNormAct) -> nn.Sequential:
    """
    Converts a LayerNormAct module to an nn.Sequential module that contains nn.LayerNorm,
    followed by dropout and activation functions.

    :param module: The module to convert.
    :return nn.Sequential: A new nn.Sequential module containing nn.LayerNorm, dropout, and activation functions.
    """
    new_norm = nn.LayerNorm(
        normalized_shape=module.normalized_shape,
        eps=module.eps,
        elementwise_affine=module.elementwise_affine,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    _copy_parameters(module, new_norm)
    new_drop = deepcopy(module.drop)
    new_act = deepcopy(module.act)
    return nn.Sequential(new_norm, new_drop, new_act)


CONVERT_FN_MAP = {
    BatchNormAct2d: _convert_batch_norm_act_2d,
    GroupNormAct: _convert_group_norm_act,
    LayerNormAct: _convert_layer_norm_act,
    Linear: _convert_linear,
}


def is_timm_custom_module(module: nn.Module):
    """
    Check that module is timm custom module and can be converted.

    :param module: The module.
    :return: `True` if module is custom module, otherwise `False`
    """
    return type(module) in list(CONVERT_FN_MAP.keys())


def convert_timm_custom_modules(module: nn.Module) -> Optional[nn.Module]:
    """
    Replaces the given module with a PyTorch native module if possible.

    :param module: The module to replace.
    :return: The replaced module if replacement is possible, None otherwise.
    """
    module_type = type(module)
    convert_fn = CONVERT_FN_MAP.get(module_type)
    if convert_fn is None:
        raise TypeError(
            f"The type of module {module_type} should be one of the following: {list(CONVERT_FN_MAP.keys())}"
        )
    return convert_fn(module)


def replace_timm_custom_modules_with_torch_native(model: nn.Module) -> nn.Module:
    """
    Replace custom module that can not be operated by NNCF to torch native modules.

    :param model: The target model.
    :return nn.Module: Transformed model.
    """
    model_copy = deepcopy(model)

    model_copy, _ = replace_modules_by_nncf_modules(
        model=model_copy,
        custom_replacer=convert_timm_custom_modules,
        predicate_fn=is_timm_custom_module,
    )

    return model_copy
