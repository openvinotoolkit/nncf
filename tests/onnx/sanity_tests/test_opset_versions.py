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

from torchvision import models
import torch
import onnx
import numpy as np
import pytest

import nncf

from tests.onnx.quantization.common import get_random_dataset_for_test
from tests.onnx.quantization.common import _get_input_key

TEST_OPSETS = [7,  # NON SUPPORTED
               10,  # PER-TENSOR ONLY
               13]  # FULLY SUPPORTED


@pytest.mark.parametrize('opset_version', TEST_OPSETS)
def test_model_opset_version(tmp_path, opset_version):
    model = models.mobilenet_v2(pretrained=True)
    input_shape = [1, 3, 224, 224]
    input_np_dtype = np.float32
    x = torch.randn(input_shape, requires_grad=False)
    torch.onnx.export(model, x, tmp_path / 'model.onnx', opset_version=opset_version)

    model = onnx.load_model(tmp_path / 'model.onnx')
    dataset = get_random_dataset_for_test(_get_input_key(model), input_shape, input_np_dtype, False)
    if opset_version == 7:
        with pytest.raises(Exception):
            _ = nncf.quantize(model, dataset, subset_size=1)
        return
    quantized_model = nncf.quantize(model, dataset, subset_size=1)
    if opset_version == 10:
        nodes_with_axis = []
        for node in filter(lambda node: node.op_type in ['QuantizeLinear', 'DequantizeLinear'],
                           quantized_model.graph.node):
            for attr in node.attribute:
                if attr.HasField('name') and 'axis' in attr.name:
                    nodes_with_axis.append(node)
        assert not nodes_with_axis
    if opset_version == 13:
        nodes_with_axis = []
        for node in filter(lambda node: node.op_type in ['QuantizeLinear', 'DequantizeLinear'],
                           quantized_model.graph.node):
            for attr in node.attribute:
                if attr.HasField('name') and 'axis' in attr.name:
                    nodes_with_axis.append(node)
        assert nodes_with_axis
