"""
 Copyright (c) 2019-2020 Intel Corporation
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

import onnx
import torch

from nncf import NNCFConfig
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import check_equal
from tests.torch.helpers import create_compressed_model_and_algo_for_test


def test_basic_model_has_expected_params():
    model = BasicConvTestModel()
    act_weights = model.conv.weight.data
    ref_weights = BasicConvTestModel.default_weight()
    act_bias = model.conv.bias.data
    ref_bias = BasicConvTestModel.default_bias()

    check_equal(act_bias, ref_bias)
    check_equal(act_weights, ref_weights)

    assert act_weights.nonzero().size(0) == model.nz_weights_num
    assert act_bias.nonzero().size(0) == model.nz_bias_num
    assert act_weights.numel() == model.weights_num
    assert act_bias.numel() == model.bias_num


def test_basic_model_is_valid():
    model = BasicConvTestModel()
    input_ = torch.ones([1, 1, 4, 4])
    ref_output = torch.ones((1, 2, 3, 3)) * (-4)
    act_output = model(input_)
    check_equal(act_output, ref_output)


def test_two_conv_model_has_expected_params():
    model = TwoConvTestModel()
    act_weights_1 = model.features[0][0].weight.data
    act_weights_2 = model.features[1][0].weight.data
    act_bias_1 = model.features[0][0].bias.data
    act_bias_2 = model.features[1][0].bias.data

    ref_weights_1 = BasicConvTestModel.default_weight()
    channel = torch.eye(3, 3).reshape([1, 1, 3, 3])
    ref_weights_2 = torch.cat((channel, channel), 1)

    check_equal(act_weights_1, ref_weights_1)
    check_equal(act_weights_2, ref_weights_2)

    check_equal(act_bias_1, BasicConvTestModel.default_bias())
    check_equal(act_bias_2, torch.tensor([0]))

    assert act_weights_1.nonzero().size(0) + act_weights_2.nonzero().size(0) == model.nz_weights_num
    assert act_bias_1.nonzero().size(0) + act_bias_2.nonzero().size(0) == model.nz_bias_num
    assert act_weights_1.numel() + act_weights_2.numel() == model.weights_num
    assert act_bias_1.numel() + act_bias_2.numel() == model.bias_num


def test_two_conv_model_is_valid():
    model = TwoConvTestModel()
    input_ = torch.ones([1, 1, 4, 4])
    ref_output = torch.tensor(-24).reshape((1, 1, 1, 1))
    act_output = model(input_)
    check_equal([act_output], [ref_output])


def load_exported_onnx_version(nncf_config: NNCFConfig, model: torch.nn.Module,
                               path_to_storage_dir) -> onnx.ModelProto:
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    onnx_checkpoint_path = path_to_storage_dir / 'model.onnx'
    compression_ctrl.export_model(onnx_checkpoint_path)
    model_proto = onnx.load_model(onnx_checkpoint_path)
    return model_proto
