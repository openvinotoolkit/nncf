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
from torch import nn

import tests.cross_fw.test_templates.helpers as helpers
from nncf.torch.function_hook.extractor import extract_model
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from nncf.torch.function_hook.wrapper import register_pre_function_hook
from nncf.torch.function_hook.wrapper import wrap_model
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import QuantizationMode
from nncf.torch.quantization.layers import SymmetricQuantizer

TEST_PARAMS = (
    (
        helpers.ConvBiasBNTestModel,
        "conv/conv2d/0",
        "bn/batch_norm/0",
    ),
    (
        helpers.ConvBNTestModel,
        "conv/conv2d/0",
        "bn/batch_norm/0",
    ),
    (
        helpers.ConvTestModel,
        "conv/conv2d/0",
        "conv/conv2d/0",
    ),
    (
        helpers.CustomConvBNTestModel,
        "conv/conv2d/0",
        "bn/batch_norm/0",
    ),
    (
        helpers.CustomConvTestModel,
        "conv/conv2d/0",
        "conv/conv2d/0",
    ),
    (
        helpers.FCTestModel,
        "fc/linear/0",
        "fc/linear/0",
    ),
)


@pytest.mark.parametrize("model_cls, input_node_name, output_node_name", TEST_PARAMS)
def test_extract_model(model_cls: type, input_node_name: str, output_node_name: str):
    example_input = torch.ones(model_cls.INPUT_SIZE)

    model: nn.Module = wrap_model(model_cls().eval())
    graph = build_nncf_graph(model, example_input)

    extracted_module = extract_model(model, graph, [input_node_name], [output_node_name])
    with torch.no_grad():
        ret1 = model(example_input)
        ret2 = extracted_module(example_input)
        assert torch.any(torch.isclose(ret1, ret2))


@pytest.mark.parametrize("model_cls, input_node_name, output_node_name", TEST_PARAMS)
def test_extract_model_for_node_with_fq(model_cls, input_node_name, output_node_name):
    example_input = torch.ones(model_cls.INPUT_SIZE)

    model = wrap_model(model_cls().eval())

    qspec = PTQuantizerSpec(
        num_bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=None,
        scale_shape=(1,),
        narrow_range=False,
        half_range=False,
        logarithm_scale=False,
    )
    fq = SymmetricQuantizer(qspec)

    register_pre_function_hook(model, input_node_name, 1, fq)

    graph = build_nncf_graph(model, example_input)

    extracted_module = extract_model(model, graph, [input_node_name], [output_node_name])
    with torch.no_grad():
        ret1 = model(example_input)
        ret2 = extracted_module(example_input)
        assert torch.all(torch.isclose(ret1, ret2))

    extracted_fn = extracted_module
    if isinstance(extracted_fn, nn.Sequential):
        extracted_fn = extracted_module[0]
