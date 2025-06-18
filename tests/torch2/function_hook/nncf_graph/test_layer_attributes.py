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

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from nncf.torch.function_hook.graph.graph_utils import TensorMeta
from nncf.torch.function_hook.nncf_graph.layer_attributes import PT2OpLayerAttributes
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from nncf.torch.function_hook.wrapper import wrap_model
from tests.torch2.function_hook.helpers import ConvModel
from tests.torch2.function_hook.helpers import MatMulLeft
from tests.torch2.function_hook.helpers import MatMulRight


@dataclass
class ParamForLayerAttributes:
    model_cls: type[nn.Module]
    node_name: str
    ref: PT2OpLayerAttributes

    def __str__(self) -> str:
        return self.model_cls.__name__


@pytest.mark.parametrize(
    "param",
    [
        ParamForLayerAttributes(
            ConvModel,
            "conv/conv2d/0",
            PT2OpLayerAttributes(
                func=torch.conv2d,
                op_args=(
                    TensorMeta(dtype=torch.float32, shape=(1, 1, 3, 3)),
                    TensorMeta(dtype=torch.float32, shape=(1, 1, 1, 1)),
                    TensorMeta(dtype=torch.float32, shape=(1,)),
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                ),
                op_kwargs={},
                constant_port_ids={1, 2},
            ),
        ),
        ParamForLayerAttributes(
            MatMulLeft,
            "/matmul/0",
            PT2OpLayerAttributes(
                func=torch.matmul,
                op_args=(
                    TensorMeta(dtype=torch.float32, shape=(1, 1)),
                    TensorMeta(dtype=torch.float32, shape=(1,)),
                ),
                op_kwargs={},
                constant_port_ids={1},
            ),
        ),
        ParamForLayerAttributes(
            MatMulRight,
            "/matmul/0",
            PT2OpLayerAttributes(
                func=torch.matmul,
                op_args=(
                    TensorMeta(dtype=torch.float32, shape=(1,)),
                    TensorMeta(dtype=torch.float32, shape=(1, 1)),
                ),
                op_kwargs={},
                constant_port_ids={0},
            ),
        ),
    ],
    ids=str,
)
def test_op_layer_attribute(param: ParamForLayerAttributes):
    # TODO(AlexanderDokuchaev): quantized model too
    model = wrap_model(param.model_cls())
    nncf_graph = build_nncf_graph(model, model.get_example_inputs())
    op_node = nncf_graph.get_node_by_name(param.node_name)
    assert op_node.layer_attributes == param.ref
