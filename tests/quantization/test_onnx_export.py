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
from tests.test_helpers import TwoConvTestModel, load_exported_onnx_version
from nncf.quantization.layers import QuantizerConfig, QuantizationMode, SymmetricQuantizer, AsymmetricQuantizer, \
    QuantizerExportMode


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


def test_onnx_export_to_fake_quantize(tmp_path):
    model = TwoConvTestModel()
    nncf_config = get_config_for_export_mode(should_be_onnx_standard=False)
    onnx_model_proto = load_exported_onnx_version(nncf_config, model,
                                                  path_to_storage_dir=tmp_path)
    num_fq = 0
    num_model_nodes = 0
    num_other_nodes = 0
    # pylint:disable=no-member
    for node in onnx_model_proto.graph.node:
        op_type = node.op_type
        if op_type == 'FakeQuantize':
            num_fq += 1
        elif op_type in ['Conv', 'Constant']:
            num_model_nodes += 1
        else:
            num_other_nodes += 1
    assert num_fq == 4
    assert num_other_nodes == 0


def test_onnx_export_to_quantize_dequantize(tmp_path):
    # It doesn't work with CPU target_device because
    # per-channel quantization is not supported in onnxruntime.
    model = TwoConvTestModel()
    nncf_config = get_config_for_export_mode(should_be_onnx_standard=True)
    nncf_config['target_device'] = 'TRIAL'
    onnx_model_proto = load_exported_onnx_version(nncf_config, model,
                                                  path_to_storage_dir=tmp_path)
    num_q = 0
    num_dq = 0
    num_model_nodes = 0
    num_other_nodes = 0
    # pylint:disable=no-member
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
    assert num_q == 4
    assert num_q == num_dq
    assert num_other_nodes == 0


def test_onnx_export_to_quantize_dequantize_per_channel():
    # SYMMETRIC
    q_config = QuantizerConfig(
        input_shape=(2, 64, 15, 10),
        bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=None,
        per_channel=True)
    sym_quantizer = SymmetricQuantizer(q_config)
    # pylint: disable=protected-access
    sym_quantizer._export_mode = QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS

    x = torch.rand((2, 64, 15, 10))
    sym_quantizer.run_export_quantization(x)

    q_config = QuantizerConfig(
        bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=None,
        per_channel=False)
    sym_quantizer = SymmetricQuantizer(q_config)
    # pylint: disable=protected-access
    sym_quantizer._export_mode = QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS

    x = torch.rand((2, 64, 15, 10))
    sym_quantizer.run_export_quantization(x)

    q_config = QuantizerConfig(
        input_shape=(2, 64, 15, 10),
        bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=None,
        per_channel=True)
    sym_quantizer = SymmetricQuantizer(q_config)
    # pylint: disable=protected-access
    sym_quantizer._export_mode = QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS
    sym_quantizer.scale = torch.nn.Parameter(torch.rand(1, 64, 1, 1))

    x = torch.rand((2, 64, 15, 10))
    try:
        sym_quantizer.run_export_quantization(x)
    except RuntimeError as e:
        assert str(
            e) == "PyTorch v1.5.0 export to ONNX using QuantizeLinear-DequantizeLinear " \
                  "doesn't support per channel quantization"
    # ASYMMETRIC
    q_config = QuantizerConfig(
        input_shape=(2, 64, 15, 10),
        bits=8,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=None,
        per_channel=True)
    assym_quantizer = AsymmetricQuantizer(q_config)
    # pylint: disable=protected-access
    assym_quantizer._export_mode = QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS

    x = torch.rand((2, 64, 15, 10))
    assym_quantizer.run_export_quantization(x)

    q_config = QuantizerConfig(
        bits=8,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=None,
        per_channel=False)
    assym_quantizer = AsymmetricQuantizer(q_config)
    # pylint: disable=protected-access
    assym_quantizer._export_mode = QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS

    x = torch.rand((2, 64, 15, 10))
    assym_quantizer.run_export_quantization(x)

    q_config = QuantizerConfig(
        input_shape=(2, 64, 15, 10),
        bits=8,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=None,
        per_channel=True)
    assym_quantizer = AsymmetricQuantizer(q_config)
    # pylint: disable=protected-access
    assym_quantizer._export_mode = QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS
    sym_quantizer.scale = torch.nn.Parameter(torch.rand(1, 64, 1, 1))

    x = torch.rand((2, 64, 15, 10))
    try:
        assym_quantizer.run_export_quantization(x)
    except RuntimeError as e:
        assert str(
            e) == "PyTorch v1.5.0 export to ONNX using QuantizeLinear-DequantizeLinear" \
                  " doesn't support per channel quantization"
