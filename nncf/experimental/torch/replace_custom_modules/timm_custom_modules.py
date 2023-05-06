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

from timm.models.layers import Linear
from timm.models.layers.norm_act import BatchNormAct2d
from timm.models.layers.norm_act import GroupNormAct
from timm.models.layers.norm_act import LayerNormAct
from torch import nn


def convert_liner(module: Linear) -> nn.Linear:
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
    new_ln.weight = deepcopy(module.weight)
    if with_bias:
        new_ln.bias = deepcopy(module.weight)
    return new_ln


def convert_batch_norm_act_2d(module: BatchNormAct2d) -> nn.Sequential:
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
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_bn.bias = deepcopy(module.weight)
    new_bn.running_mean = deepcopy(module.running_mean)
    new_bn.running_var = deepcopy(module.running_var)
    new_bn.weight = deepcopy(module.weight)

    new_drop = deepcopy(module.drop)
    new_act = deepcopy(module.act)
    return nn.Sequential(new_bn, new_drop, new_act)


def convert_group_norm_act(module: GroupNormAct) -> nn.Sequential:
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
    new_gn.bias = deepcopy(module.weight)
    new_gn.weight = deepcopy(module.weight)
    new_drop = deepcopy(module.drop)
    new_act = deepcopy(module.act)
    return nn.Sequential(new_gn, new_drop, new_act)


def convert_layer_norm_act(module: LayerNormAct) -> nn.Sequential:
    """
    Converts a LayerNormAct module to an nn.Sequential module that contains nn.LayerNorm,
    followed by dropout and activation functions.

    :param module: The module to convert.
    :return nn.Sequential: A new nn.Sequential module containing nn.LayerNorm, dropout, and activation functions.
    """
    with_bias = module.bias is not None
    new_norm = nn.LayerNorm(
        normalized_shape=module.normalized_shape,
        eps=module.normalized_shape,
        elementwise_affine=module.elementwise_affine,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_norm.weight = deepcopy(module.weight)
    if with_bias:
        new_norm.bias = deepcopy(module.weight)
    new_drop = deepcopy(module.drop)
    new_act = deepcopy(module.act)
    return nn.Sequential(new_norm, new_drop, new_act)


REPLACE_FN_MAP = {
    BatchNormAct2d: convert_batch_norm_act_2d,
    GroupNormAct: convert_group_norm_act,
    LayerNormAct: convert_layer_norm_act,
    Linear: convert_liner,
}


def replace_timm_modules(module: nn.Module):
    """
    Replaces the given module with a PyTorch native module if possible.

    :param module: The module to replace.
    :return: The replaced module if replacement is possible, None otherwise.
    """
    convert_fn = REPLACE_FN_MAP.get(type(module))
    if convert_fn is None:
        return None
    return convert_fn(module)
