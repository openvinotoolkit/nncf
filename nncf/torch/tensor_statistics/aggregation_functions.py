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

import torch

from nncf.common.utils.registry import Registry


aggregator = Registry('PTAggregationFunctions')


def get_aggregation_function(name):
    return aggregator.get(name)


def convert_from_numpy_rs_to_torch_rs(x, reduction_shape_np):
    nncf_torch_shape = []
    for i in range(x.dim()):
        if i in reduction_shape_np:
            nncf_torch_shape.append(1)
        else:
            nncf_torch_shape.append(x.shape[i])
    if all(item == 1 for item in nncf_torch_shape):
        nncf_torch_shape = [1]
    return tuple(nncf_torch_shape)


@aggregator.register()
def pt_reduce_min(x, reduction_shape):
    new = torch.amin(x, dim=reduction_shape)
    nncf_torch_shape = convert_from_numpy_rs_to_torch_rs(x, reduction_shape)
    return new.view(nncf_torch_shape)


@aggregator.register()
def pt_reduce_max(x, reduction_shape):
    new = torch.amax(x, dim=reduction_shape)
    nncf_torch_shape = convert_from_numpy_rs_to_torch_rs(x, reduction_shape)
    return new.view(nncf_torch_shape)


@aggregator.register()
def pt_abs(x):
    return torch.abs(x)


@aggregator.register()
def pt_min(x1, x2):
    return torch.min(x1, x2)


@aggregator.register()
def pt_max(x1, x2):
    return torch.max(x1, x2)


@aggregator.register()
def pt_tensor_min(x, axis):
    val, _ = x.min(dim=axis)
    return val


@aggregator.register()
def pt_tensor_max(x, axis):
    val, _ = x.max(dim=axis)
    return val


@aggregator.register()
def pt_mean(x, axis):
    return x.mean(dim=axis)


@aggregator.register()
def pt_stack(x):
    return torch.stack(tuple(x))


def convert_shape(shape):
    if all(dim == 1 for dim in shape[1:]):
        return [1]
    return [1] + shape[1:]


@aggregator.register()
def pt_list_to_extend_stat_history(x):
    return [t.view(convert_shape(list(x.size()))) for t in torch.unbind(x)]
