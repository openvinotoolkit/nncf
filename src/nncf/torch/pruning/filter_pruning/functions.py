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
import torch


def l1_filter_norm(weight_tensor, dim=0):
    """
    Calculates L1 for weight_tensor for the selected dimension.
    """
    weight_tensor = weight_tensor.transpose(0, dim).contiguous()
    return torch.norm(weight_tensor.view(weight_tensor.shape[0], -1), p=1, dim=1)


def l2_filter_norm(weight_tensor, dim=0):
    """
    Calculates L2 for weight_tensor for the selected dimension.
    """
    weight_tensor = weight_tensor.transpose(0, dim).contiguous()
    return torch.norm(weight_tensor.view(weight_tensor.shape[0], -1), p=2, dim=1)


def tensor_l2_normalizer(weight_tensor):
    norm = torch.sqrt(torch.sum(torch.abs(weight_tensor) ** 2))
    return weight_tensor / norm


def geometric_median_filter_norm(weight_tensor, dim=0):
    """
    Compute geometric median norm for filters.
    :param weight_tensor: tensor with weights
    :param dim: dimension of output channel
    :return: metric value for every weight from weights_tensor
    """
    weight_tensor = weight_tensor.transpose(0, dim).contiguous()
    filters_count = weight_tensor.size(0)
    weight_vec = weight_tensor.view(filters_count, -1)
    similarity_matrix = torch.cdist(weight_vec[None, :], weight_vec[None, :], p=2.0)
    return similarity_matrix.squeeze().sum(axis=0).to(weight_tensor.device)


FILTER_IMPORTANCE_FUNCTIONS = {
    "L2": l2_filter_norm,
    "L1": l1_filter_norm,
    "geometric_median": geometric_median_filter_norm,
}


def calculate_binary_mask(importance, threshold):
    return (importance >= threshold).float()
