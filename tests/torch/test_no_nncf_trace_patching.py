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
from torch import nn
from torch.nn import functional as F

from nncf.torch import create_compressed_model
from nncf.torch import disable_tracing
from nncf.torch.dynamic_graph.context import get_current_context
from tests.torch.helpers import create_conv
from tests.torch.helpers import get_empty_config


class SimpleModel(nn.Module):
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
        for _ in range(torch.greater(x, 0.5).sum()):
            output = output + x
        return output


def test_no_trace_model_patching():
    config = get_empty_config()
    config["input_info"] = {"sample_size": [1, 1, 4, 4], "filler": "random"}

    # Not patching anything: all output nodes are traced
    _, compressed_model = create_compressed_model(SimpleModel(True), config)
    assert len(compressed_model.nncf.get_original_graph().get_output_nodes()) == 2

    # Patching a function results with no_nncf_trace in method not producing an output node
    disable_tracing(SimpleModel.ambiguous_op)
    _, compressed_model = create_compressed_model(SimpleModel(False), get_empty_config())
    assert len(compressed_model.nncf.get_original_graph().get_output_nodes()) == 1
