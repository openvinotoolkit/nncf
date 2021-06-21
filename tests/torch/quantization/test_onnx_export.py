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
from itertools import product

import pytest
import torch

from nncf import NNCFConfig
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import QuantizationMode
from nncf.torch.quantization.layers import QuantizerExportMode
from tests.torch.test_helpers import TwoConvTestModel
from tests.torch.test_helpers import load_exported_onnx_version
from tests.torch.helpers import  register_bn_adaptation_init_args


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
    register_bn_adaptation_init_args(nncf_config)
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


INPUT_TENSOR_SHAPE = (2, 64, 15, 10)
PER_CHANNEL_AQ_SCALE_SHAPE = (1, INPUT_TENSOR_SHAPE[1], 1, 1)


@pytest.mark.parametrize('per_channel, qmode, export_mode',
                         product(
                             [True, False],
                             [QuantizationMode.SYMMETRIC, QuantizationMode.ASYMMETRIC],
                             [QuantizerExportMode.FAKE_QUANTIZE, QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS]
                             ))
def test_onnx_export_to_quantize_dequantize_per_channel(per_channel: bool,
                                                        qmode: QuantizationMode,
                                                        export_mode: QuantizerExportMode):
    scale_shape = PER_CHANNEL_AQ_SCALE_SHAPE if per_channel else (1,)
    qspec = PTQuantizerSpec(
        scale_shape=scale_shape,
        num_bits=8,
        mode=qmode,
        signedness_to_force=None,
        logarithm_scale=False,
        narrow_range=False,
        half_range=False,
    )

    q_cls = QUANTIZATION_MODULES.get(qmode)
    quantizer = q_cls(qspec)
    if qmode is QuantizationMode.SYMMETRIC:
        quantizer.scale = torch.nn.Parameter(torch.rand_like(quantizer.scale))
    else:
        quantizer.input_low = torch.nn.Parameter(torch.rand_like(quantizer.input_low))
        quantizer.input_range = torch.nn.Parameter(torch.rand_like(quantizer.input_range))
    # pylint: disable=protected-access
    quantizer._export_mode = export_mode

    x = torch.rand(INPUT_TENSOR_SHAPE)
    if quantizer.per_channel and export_mode is QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS:
        with pytest.raises(RuntimeError):
            quantizer.run_export_quantization(x)
    else:
        quantizer.run_export_quantization(x)
