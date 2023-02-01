"""
 Copyright (c) 2023 Intel Corporation
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
from torch import nn
from torch.nn import functional as F

from nncf.common.utils.patcher import PATCHER
from nncf.torch import create_compressed_model
from nncf.torch.dynamic_graph.context import get_current_context
from nncf.torch.dynamic_graph.context import patch_method_with_no_nncf_trace
from tests.torch.helpers import create_conv
from tests.torch.helpers import get_empty_config


CORRECT_WRAPPER_STACK = "base"


def wrapper1(self, fn, *args, **kwargs):  # pylint: disable=unused-argument
    kwargs["wrapper_stack"] += "_wrapper1"
    return fn(*args, **kwargs)


def wrapper2(self, fn, *args, **kwargs):  # pylint: disable=unused-argument
    kwargs["wrapper_stack"] += "_wrapper2"
    return fn(*args, **kwargs)


def assert_wrapper_stack(wrapper_stack=None):
    assert wrapper_stack == CORRECT_WRAPPER_STACK


class TestOverrideClass:
    def assert_wrapper_stack_method(self, wrapper_stack=None):
        assert wrapper_stack == CORRECT_WRAPPER_STACK

    @staticmethod
    def assert_wrapper_stack_static(wrapper_stack=None):
        assert wrapper_stack == CORRECT_WRAPPER_STACK

    @classmethod
    def assert_wrapper_stack_class(cls, wrapper_stack=None):
        assert wrapper_stack == CORRECT_WRAPPER_STACK


def test_patcher():
    global CORRECT_WRAPPER_STACK

    test_obj = TestOverrideClass()

    # Test without patching
    assert_wrapper_stack(wrapper_stack="base")
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")
    TestOverrideClass.assert_wrapper_stack_static(wrapper_stack="base")
    TestOverrideClass.assert_wrapper_stack_class(wrapper_stack="base")

    # Test non-class method
    PATCHER.patch(assert_wrapper_stack, wrapper1)
    CORRECT_WRAPPER_STACK = "base_wrapper1"
    assert_wrapper_stack(wrapper_stack="base")

    # Test single patch static method
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_static, wrapper1)
    CORRECT_WRAPPER_STACK = "base_wrapper1"
    test_obj.assert_wrapper_stack_static(wrapper_stack="base")  # doesn't work if called from class

    # Test single patch class method
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_class, wrapper1)
    CORRECT_WRAPPER_STACK = "base_wrapper1"
    test_obj.assert_wrapper_stack_class(wrapper_stack="base")  # doesn't work if called from class

    # Test single patch object method
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_method, wrapper1)
    CORRECT_WRAPPER_STACK = "base_wrapper1"
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")
    once_wrapped_ref = TestOverrideClass.assert_wrapper_stack_method

    # Test applying two nested patches
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_method, wrapper2, override=False)
    CORRECT_WRAPPER_STACK = "base_wrapper2_wrapper1"
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")  # doesn't work if patched thrice

    # Test overriding patch
    TestOverrideClass.assert_wrapper_stack_method = once_wrapped_ref   # revert the last patching
    PATCHER.patch(TestOverrideClass.assert_wrapper_stack_method, wrapper1, override=True)
    CORRECT_WRAPPER_STACK = "base_wrapper1"
    test_obj.assert_wrapper_stack_method(wrapper_stack="base")


class TestModel(nn.Module):
    """
    A test model with an operation resulting in an ambiguous graph.
    Ambiguous operation output is put into the model output for testing convenience.
    """

    def __init__(self, correct_is_tracing):
        super().__init__()
        self.conv = create_conv(in_channels=1, out_channels=1, kernel_size=2)
        self.correct_is_tracing = correct_is_tracing

    def forward(self, x):
        x = F.sigmoid(self.conv(x - 0.5))
        output = self.ambiguous_op(x)
        return x, output

    def ambiguous_op(self, x):
        assert get_current_context().is_tracing == self.correct_is_tracing

        output = torch.zeros_like(x)
        for idx in range(torch.greater(x, 0.5).sum()):
            output = output + x
        return output


def test_no_trace_model_patching():
    config = get_empty_config()
    config["input_info"] = {
        "sample_size": [1, 1, 4, 4],
        "filler": "random"
    }

    # Not patching anything: all output nodes are traced
    _, compressed_model = create_compressed_model(TestModel(True), config)
    assert len(compressed_model._original_graph._output_nncf_nodes) == 2

    # Patching a function results with no_nncf_trace in method not producing an output node
    patch_method_with_no_nncf_trace(TestModel.ambiguous_op)
    _, compressed_model = create_compressed_model(TestModel(False), get_empty_config())
    assert len(compressed_model._original_graph._output_nncf_nodes) == 1
