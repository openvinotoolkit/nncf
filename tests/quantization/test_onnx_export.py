"""
 Copyright (c) 2020 Intel Corporation
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
from nncf import NNCFConfig
from tests.helpers import TwoConvTestModel, create_compressed_model_and_algo_for_test
import onnx


def get_config_for_export_mode(should_be_onnx_standard: bool) -> NNCFConfig:
    nncf_config = NNCFConfig()
    nncf_config.update({
        "input_info": {
            "sample_size": [1, 1, 4, 4]
        },
        "compression": {
            "algorithm": "quantization",
            "export_to_onnx_standard_ops": should_be_onnx_standard
        }
    })
    return nncf_config


def load_exported_onnx_version(model: torch.nn.Module, should_be_onnx_standard: bool,
                               path_to_storage_dir) -> onnx.ModelProto:
    nncf_config = get_config_for_export_mode(should_be_onnx_standard)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    onnx_checkpoint_path = path_to_storage_dir / 'model.onnx'
    compression_ctrl.export_model(onnx_checkpoint_path)
    model_proto = onnx.load_model(onnx_checkpoint_path)
    return model_proto


def test_onnx_export_to_fake_quantize(tmp_path):
    model = TwoConvTestModel()
    onnx_model_proto = load_exported_onnx_version(model,
                                                  should_be_onnx_standard=False,
                                                  path_to_storage_dir=tmp_path)
    num_fq = 0
    num_model_nodes = 0
    num_other_nodes = 0
    #pylint:disable=no-member
    for node in onnx_model_proto.graph.node:
        op_type = node.op_type
        if op_type == 'FakeQuantize':
            num_fq += 1
        elif op_type in ['Conv', 'Constant']:
            num_model_nodes += 1
        else:
            num_other_nodes += 1
    assert num_fq > 0
    assert num_other_nodes == 0


def test_onnx_export_to_quantize_dequantize(tmp_path):
    model = TwoConvTestModel()
    onnx_model_proto = load_exported_onnx_version(model,
                                                  should_be_onnx_standard=True,
                                                  path_to_storage_dir=tmp_path)
    num_q = 0
    num_dq = 0
    num_model_nodes = 0
    num_other_nodes = 0
    #pylint:disable=no-member
    for node in onnx_model_proto.graph.node:
        op_type = node.op_type
        if op_type == 'QuantizeLinear':
            num_q += 1
        elif op_type == 'DequantizeLinear':
            num_dq += 1
        elif op_type in ['Conv', 'Constant']:
            num_model_nodes += 1
        else:
            num_other_nodes += 1
    assert num_q > 0
    assert num_q == num_dq
    assert num_other_nodes == 0
