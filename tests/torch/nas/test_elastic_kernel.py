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

import pytest
import torch
from torch.nn import Parameter

from tests.torch.nas.creators import create_single_conv_kernel_supernet
from tests.torch.nas.helpers import do_conv2d
from tests.torch.nas.helpers import ref_kernel_transform

BASIC_ELASTIC_KERNEL_PARAMS = {"max_num_kernels": 2}


###########################
# Behavior
###########################


def test_elastic_kernel_with_odd_value():
    with pytest.raises(AssertionError):
        create_single_conv_kernel_supernet(kernel_size=2)


def test_elastic_kernel_with_less_than_minimum_value():
    kernel_handler, _ = create_single_conv_kernel_supernet()
    with pytest.raises(ValueError):
        kernel_handler.activate_subnet_for_config([1])


def test_elastic_kernel_with_more_than_maximum_value():
    kernel_handler, _ = create_single_conv_kernel_supernet()
    with pytest.raises(ValueError):
        kernel_handler.activate_subnet_for_config([9])


###########################
# Output checking
###########################


def test_elastic_kernel_with_maximum_value():
    _, supernet = create_single_conv_kernel_supernet()
    device = next(iter(supernet.parameters())).device
    input_ = torch.ones([1, 1, 5, 5]).to(device)
    conv = supernet.conv
    actual_output = supernet(input_)

    ref_padding = 2
    ref_output = do_conv2d(conv, input_, padding=ref_padding)

    assert torch.equal(actual_output, ref_output)


def test_elastic_kernel_with_intermediate_value():
    kernel_handler, supernet = create_single_conv_kernel_supernet()

    device = next(iter(supernet.parameters())).device
    input_ = torch.ones([1, 1, 5, 5]).to(device)
    conv = supernet.conv
    kernel_handler.activate_subnet_for_config([3])
    actual_output = supernet(input_)

    ref_padding = 1
    ref_weights = conv.weight[:, :, 1:4, 1:4]
    ref_output = do_conv2d(conv, input_, padding=ref_padding, weight=ref_weights)

    assert torch.equal(actual_output, ref_output)


def test_elastic_kernel_output_shape():
    kernel_handler, supernet = create_single_conv_kernel_supernet(kernel_size=9, padding=2)

    device = next(iter(supernet.parameters())).device
    input_ = torch.ones([1, 1, 9, 9]).to(device)

    original_model = supernet.nncf.get_clean_shallow_copy()
    ref_output = original_model(input_)

    kernel_size_list = [9, 7, 5]
    for kernel_size in kernel_size_list:
        kernel_handler.activate_subnet_for_config([kernel_size])
        actual_output = supernet(input_)
        assert actual_output.shape == ref_output.shape


def test_elastic_kernel_with_custom_transition_matrix():
    kernel_handler, supernet = create_single_conv_kernel_supernet()
    device = next(iter(supernet.parameters())).device
    custom_transition_matrix = torch.ones([3**2, 3**2]).to(device)

    elastic_kernel_op = kernel_handler._elastic_kernel_ops[0]
    elastic_kernel_op.__setattr__(f"{5}to{3}_matrix", Parameter(custom_transition_matrix))
    input_ = torch.ones([1, 1, 5, 5]).to(device)
    conv = supernet.conv

    kernel_handler.activate_subnet_for_config([3])
    actual_output = supernet(input_)

    ref_padding = 1
    ref_weights = ref_kernel_transform(conv.weight, transition_matrix=custom_transition_matrix)
    ref_output = do_conv2d(conv, input_, padding=ref_padding, weight=ref_weights)
    assert torch.equal(actual_output, ref_output)
