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
from nncf.common.graph.transformations.commands import TargetType
from nncf.torch import wrap_model
from nncf.torch.extractor import extract_model
from nncf.torch.graph.transformations.command_creation import create_quantizer_insertion_command
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.model_transformer import PTTransformationLayout
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import QuantizationMode
from nncf.torch.quantization.layers import SymmetricQuantizer


@pytest.mark.parametrize(
    "model_cls, input_node_name, output_node_name",
    (
        (
            helpers.ConvBiasBNTestModel,
            "ConvBiasBNTestModel/Conv2d[conv]/conv2d_0",
            "ConvBiasBNTestModel/BatchNorm2d[bn]/batch_norm_0",
        ),
        (
            helpers.ConvBNTestModel,
            "ConvBNTestModel/Conv2d[conv]/conv2d_0",
            "ConvBNTestModel/BatchNorm2d[bn]/batch_norm_0",
        ),
        (
            helpers.ConvTestModel,
            "ConvTestModel/Conv2d[conv]/conv2d_0",
            "ConvTestModel/Conv2d[conv]/conv2d_0",
        ),
        (
            helpers.CustomConvBNTestModel,
            "CustomConvBNTestModel/CustomConv[conv]/conv2d_0",
            "CustomConvBNTestModel/CustomBN2d[bn]/batch_norm_0",
        ),
        (
            helpers.CustomConvTestModel,
            "CustomConvTestModel/CustomConv[conv]/conv2d_0",
            "CustomConvTestModel/CustomConv[conv]/conv2d_0",
        ),
    ),
)
def test_extract_model(model_cls, input_node_name, output_node_name):
    example_input = torch.ones(model_cls.INPUT_SIZE)

    model = wrap_model(model_cls().eval(), example_input=example_input, trace_parameters=True)
    extracted_module = extract_model(model, [input_node_name], [output_node_name])
    with torch.no_grad():
        ret1 = model(example_input)
        ret2 = extracted_module(example_input)
        assert torch.any(torch.isclose(ret1, ret2))


@pytest.mark.parametrize(
    "model_cls, input_node_name, output_node_name",
    (
        (
            helpers.ConvBiasBNTestModel,
            "ConvBiasBNTestModel/Conv2d[conv]/conv2d_0",
            "ConvBiasBNTestModel/BatchNorm2d[bn]/batch_norm_0",
        ),
        (
            helpers.ConvBNTestModel,
            "ConvBNTestModel/Conv2d[conv]/conv2d_0",
            "ConvBNTestModel/BatchNorm2d[bn]/batch_norm_0",
        ),
        (
            helpers.ConvTestModel,
            "ConvTestModel/Conv2d[conv]/conv2d_0",
            "ConvTestModel/Conv2d[conv]/conv2d_0",
        ),
        (
            helpers.CustomConvBNTestModel,
            "CustomConvBNTestModel/CustomConv[conv]/conv2d_0",
            "CustomConvBNTestModel/CustomBN2d[bn]/batch_norm_0",
        ),
        (
            helpers.CustomConvTestModel,
            "CustomConvTestModel/CustomConv[conv]/conv2d_0",
            "CustomConvTestModel/CustomConv[conv]/conv2d_0",
        ),
    ),
)
def test_extract_model_for_node_with_fq(model_cls, input_node_name, output_node_name):
    example_input = torch.ones(model_cls.INPUT_SIZE)

    model = wrap_model(model_cls().eval(), example_input=example_input, trace_parameters=True)

    transformer = PTModelTransformer(model)
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
    command = create_quantizer_insertion_command(
        PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, input_node_name, input_port_id=1), fq
    )
    layout = PTTransformationLayout()
    layout.register(command)
    q_model = transformer.transform(layout)

    extracted_module = extract_model(model, [input_node_name], [output_node_name])
    with torch.no_grad():
        ret1 = q_model(example_input)
        ret2 = extracted_module(example_input)
        assert torch.all(torch.isclose(ret1, ret2))

    extracted_fn = extracted_module
    if isinstance(extracted_fn, nn.Sequential):
        extracted_fn = extracted_module[0]

    assert extracted_fn.fn_name is not None
