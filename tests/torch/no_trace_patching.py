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
from nncf.torch.dynamic_graph.context import no_nncf_trace
from tests.torch.helpers import create_conv
from tests.torch.helpers import get_empty_config


class TestModel(nn.Module):
    def __init__(self, is_tracing):
        super().__init__()
        self.conv = create_conv(in_channels=1, out_channels=1, kernel_size=2)
        self.last_call_was_tracing = None
        self.is_tracing = is_tracing

    def forward(self, x):
        output1 = F.sigmoid(self.conv(x))
        output2 = self.to_indices(output1)
        return output1, output2

    def to_indices(self, x):
        assert get_current_context().is_tracing == self.is_tracing
        res = torch.nonzero(x > 0.5)
        return res


def no_nncf_trace_wrapper(self, fn, *args, **kwargs):  # pylint: disable=unused-argument
    with no_nncf_trace():
        return fn(*args, **kwargs)


def test_monkey_patch_model_method():
    sample_input = torch.rand([1, 1, 4, 4]) - 0.5

    # Not patching TestModel.to_indices(): torch.nonzero node is traced
    model = TestModel(is_tracing=True)
    compression_ctrl, compressed_model = create_compressed_model(model, get_empty_config())
    compressed_model(sample_input)
    assert len(compression_ctrl.model._original_graph._output_nncf_nodes) == 2

    # Patching TestModel.to_indices(): torch.nonzero is not traced
    PATCHER.patch(TestModel.to_indices, no_nncf_trace_wrapper)
    model = TestModel(is_tracing=False)
    compression_ctrl, compressed_model = create_compressed_model(model, get_empty_config())
    compressed_model(sample_input)
    assert len(compression_ctrl.model._original_graph._output_nncf_nodes) == 1
