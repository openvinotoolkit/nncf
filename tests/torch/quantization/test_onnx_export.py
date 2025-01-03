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
from typing import Dict, List, Tuple

import numpy as np
import onnx
import pytest
import torch
from torch import nn

from nncf import NNCFConfig
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import QuantizerExportMode
from nncf.torch.quantization.layers import SymmetricQuantizer
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_all_inputs_for_graph_node
from tests.torch.helpers import get_nodes_by_type
from tests.torch.helpers import load_exported_onnx_version
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.helpers import resolve_constant_node_inputs_to_values


def get_config_for_export_mode(should_be_onnx_standard: bool) -> NNCFConfig:
    nncf_config = NNCFConfig()
    nncf_config.update(
        {
            "input_info": {"sample_size": [1, 1, 4, 4]},
            "compression": {"algorithm": "quantization", "export_to_onnx_standard_ops": should_be_onnx_standard},
        }
    )
    register_bn_adaptation_init_args(nncf_config)
    return nncf_config


def test_onnx_export_to_fake_quantize(tmp_path):
    model = TwoConvTestModel()
    nncf_config = get_config_for_export_mode(should_be_onnx_standard=False)
    onnx_model_proto = load_exported_onnx_version(nncf_config, model, path_to_storage_dir=tmp_path)
    num_fq = 0
    num_model_nodes = 0
    num_other_nodes = 0

    for node in onnx_model_proto.graph.node:
        op_type = node.op_type
        if op_type == "FakeQuantize":
            num_fq += 1
        elif op_type in ["Conv", "Constant"]:
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
    nncf_config["target_device"] = "TRIAL"
    onnx_model_proto = load_exported_onnx_version(
        nncf_config, model, path_to_storage_dir=tmp_path, save_format="onnx_13"
    )
    num_q = 0
    num_dq = 0
    num_model_nodes = 0
    num_other_nodes = 0

    for node in onnx_model_proto.graph.node:
        op_type = node.op_type
        if op_type == "QuantizeLinear":
            num_q += 1
        elif op_type == "DequantizeLinear":
            num_dq += 1
        elif op_type in ["Conv", "Constant"]:
            num_model_nodes += 1
        else:
            num_other_nodes += 1
    assert num_q == 4
    assert num_q == num_dq
    assert num_other_nodes == 0


INPUT_TENSOR_SHAPE = (2, 64, 15, 10)
PER_CHANNEL_AQ_SCALE_SHAPE = (1, INPUT_TENSOR_SHAPE[1], 1, 1)


@pytest.mark.parametrize(
    "export_mode", (QuantizerExportMode.FAKE_QUANTIZE, QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS)
)
def test_onnx_export_to_quantize_dequantize_per_channel(
    is_per_channel: bool, quantization_mode: QuantizationMode, export_mode: QuantizerExportMode
):
    scale_shape = PER_CHANNEL_AQ_SCALE_SHAPE if is_per_channel else (1,)
    qspec = PTQuantizerSpec(
        scale_shape=scale_shape,
        num_bits=8,
        mode=quantization_mode,
        signedness_to_force=None,
        logarithm_scale=False,
        narrow_range=False,
        half_range=False,
        is_quantized_on_export=False,
    )

    q_cls = QUANTIZATION_MODULES.get(quantization_mode)
    quantizer = q_cls(qspec)
    if quantization_mode is QuantizationMode.SYMMETRIC:
        quantizer.scale = torch.nn.Parameter(torch.rand_like(quantizer.scale))
    else:
        quantizer.input_low = torch.nn.Parameter(torch.rand_like(quantizer.input_low))
        quantizer.input_range = torch.nn.Parameter(torch.rand_like(quantizer.input_range))

    quantizer._export_mode = export_mode

    x = torch.rand(INPUT_TENSOR_SHAPE)
    quantizer.run_export_quantization(x)


class TargetCompressionIdxTestModel(torch.nn.Module):
    CONV2D_TARGET_CHANNEL_COUNT = 5
    CONV2D_TRANSPOSE_TARGET_CHANNEL_COUNT = 10

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=self.CONV2D_TARGET_CHANNEL_COUNT, kernel_size=(1, 1))
        self.conv_t = torch.nn.ConvTranspose2d(
            in_channels=self.CONV2D_TARGET_CHANNEL_COUNT,
            out_channels=self.CONV2D_TRANSPOSE_TARGET_CHANNEL_COUNT,
            kernel_size=(1, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_t(x)
        return x


def get_weight_fq_for_conv_node(node: onnx.NodeProto, graph: onnx.GraphProto):
    weight_input_tensor_id = node.input[1]
    matches = [x for x in graph.node if weight_input_tensor_id in x.output]
    assert len(matches) == 1
    match = next(iter(matches))
    assert match.op_type == "FakeQuantize"
    return match


def get_input_low_input_high_for_wfq_node(
    wfq_node: onnx.NodeProto, graph: onnx.GraphProto
) -> Tuple[onnx.AttributeProto, onnx.AttributeProto]:
    assert wfq_node.op_type == "FakeQuantize"
    conv_wfq_inputs = list(resolve_constant_node_inputs_to_values(wfq_node, graph).values())
    return conv_wfq_inputs[1], conv_wfq_inputs[2]


def test_target_compression_idx(tmp_path):
    model = TargetCompressionIdxTestModel()
    nncf_config = get_config_for_export_mode(should_be_onnx_standard=False)
    onnx_model_proto = load_exported_onnx_version(nncf_config, model, path_to_storage_dir=tmp_path)
    onnx_graph = onnx_model_proto.graph
    conv_nodes = get_nodes_by_type(onnx_model_proto, "Conv")
    assert len(conv_nodes) == 1
    conv_node = next(iter(conv_nodes))
    conv_wfq_node = get_weight_fq_for_conv_node(conv_node, onnx_graph)
    input_low_attr, input_high_attr = get_input_low_input_high_for_wfq_node(conv_wfq_node, onnx_graph)
    assert input_low_attr.shape == (TargetCompressionIdxTestModel.CONV2D_TARGET_CHANNEL_COUNT, 1, 1, 1)
    assert input_low_attr.shape == input_high_attr.shape

    conv_t_nodes = get_nodes_by_type(onnx_model_proto, "ConvTranspose")
    assert len(conv_t_nodes) == 1
    conv_t_node = next(iter(conv_t_nodes))
    conv_t_wfq_node = get_weight_fq_for_conv_node(conv_t_node, onnx_graph)
    input_low_t_attr, input_high_t_attr = get_input_low_input_high_for_wfq_node(conv_t_wfq_node, onnx_graph)
    assert input_low_t_attr.shape == (1, TargetCompressionIdxTestModel.CONV2D_TRANSPOSE_TARGET_CHANNEL_COUNT, 1, 1)
    assert input_low_t_attr.shape == input_high_t_attr.shape


class ModelWithBranches(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(2, 2, (1, 1))
        self.conv_2 = torch.nn.Conv2d(2, 2, (1, 1), groups=2)
        self.conv_3 = torch.nn.Conv2d(2, 2, (1, 1), groups=2)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = x + x
        return x1, x2, x3, x4


def get_successors(node: onnx.NodeProto, graph: onnx.GraphProto) -> List[onnx.NodeProto]:
    retval = []
    for output_name in node.output:
        for target_node in graph.node:
            if output_name in target_node.input:
                retval.append(target_node)
    return retval


@pytest.mark.parametrize(
    "export_mode", [QuantizerExportMode.FAKE_QUANTIZE, QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS]
)
def test_branching_fqs_are_not_chained(tmp_path, export_mode):
    nncf_config = NNCFConfig.from_dict(
        {
            "input_info": {"sample_size": [1, 2, 2, 2]},
            "compression": {
                "algorithm": "quantization",
                "preset": "mixed",
                "ignored_scopes": ["/nncf_model_input_0", "{re}.*__add__.*"],
                "initializer": {
                    "range": {"num_init_samples": 0},
                    "batchnorm_adaptation": {"num_bn_adaptation_samples": 0},
                },
            },
        }
    )
    onnx_model_proto = load_exported_onnx_version(nncf_config, ModelWithBranches(), path_to_storage_dir=tmp_path)
    target_node_type = "FakeQuantize" if export_mode is QuantizerExportMode.FAKE_QUANTIZE else "DequantizeLinear"
    quantizer_nodes = get_nodes_by_type(onnx_model_proto, target_node_type)
    # Quantizer nodes should, for this model, immediately be followed by the quantized operation. Chained quantizers
    # mean that the ONNX export was incorrect.

    follower_node_lists = [get_successors(x, onnx_model_proto.graph) for x in quantizer_nodes]
    follower_nodes = []
    for lst in follower_node_lists:
        follower_nodes += lst
    follower_node_types = [x.op_type for x in follower_nodes]
    assert not any(x == target_node_type for x in follower_node_types)


def set_parameters_to_quantizer_and_get_attrs(
    quantizer: BaseQuantizer, paramaters_to_set: Dict
) -> Tuple[np.ndarray, np.ndarray, int]:
    if isinstance(quantizer, SymmetricQuantizer):
        return set_scale_to_sym_quantizer_and_get_attrs(quantizer, **paramaters_to_set)
    return set_input_low_and_input_range_to_asym_quantizer_and_get_attrs(quantizer, **paramaters_to_set)


def set_scale_to_sym_quantizer_and_get_attrs(
    quantizer: SymmetricQuantizer, scale: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    scale = np.full(quantizer.scale.size(), scale)
    levels = quantizer.levels
    level_low = quantizer.level_low
    level_high = quantizer.level_high
    input_low = scale * (level_low / level_high)
    input_range = scale - input_low
    quant_len = input_range / (levels - 1)
    quantizer.scale = nn.Parameter(torch.from_numpy(scale.astype(np.single)))
    return input_low, quant_len, levels


def set_input_low_and_input_range_to_asym_quantizer_and_get_attrs(
    quantizer: AsymmetricQuantizer, input_low: float, input_range: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    input_low = np.full(quantizer.input_low.size(), input_low)
    input_range = np.full(quantizer.input_low.size(), input_range)
    levels = quantizer.levels
    quant_len = input_range / (levels - 1)
    quantizer.input_low = nn.Parameter(torch.from_numpy(input_low.astype(np.single)))
    quantizer.input_range = nn.Parameter(torch.from_numpy(input_range.astype(np.single)))
    return input_low, quant_len, levels


def generate_middle_quants(
    size: List[int], input_low: np.ndarray, quant_len: np.ndarray, levels: np.ndarray
) -> torch.Tensor:
    ref_weights = [input_low + (i + 0.5) * quant_len for i in range(levels)]
    elems = np.prod(size)
    ref_weights = ref_weights * int(np.round(0.5 + elems / levels))
    ref_weights = np.reshape(np.array(ref_weights).flatten()[:elems], size, "F")
    return torch.from_numpy(ref_weights.astype(np.single))


@pytest.mark.parametrize(
    "quantization_mode, parameters_to_set",
    [("symmetric", {"scale": 1.0}), ("asymmetric", {"input_low": -1.0, "input_range": 3.0})],
)
def test_export_quantized_weights_with_middle_quants(tmp_path, is_half_range, quantization_mode, parameters_to_set):
    model = TwoConvTestModel()
    sample_size = [1, 1, 20, 20]
    config = get_config_for_export_mode(False)
    config["compression"]["weights"] = {"mode": quantization_mode}
    if not is_half_range:
        config["compression"]["overflow_fix"] = "disable"
    config["input_info"]["sample_size"] = sample_size

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    quantizers = compression_ctrl.weight_quantizers.values()
    for quantizer in quantizers:
        input_low, quant_len, levels = set_parameters_to_quantizer_and_get_attrs(
            quantizer.quantizer_module_ref, parameters_to_set
        )
        ref_weights = generate_middle_quants(
            list(quantizer.quantized_module.weight.size()), input_low, quant_len, levels
        )
        quantizer.quantized_module.weight = nn.Parameter(ref_weights)

    onnx_checkpoint_path = str(tmp_path / "two_conv_model_int8.onnx")
    compression_ctrl.export_model(onnx_checkpoint_path)
    model_onnx = onnx.load(onnx_checkpoint_path)

    fq_nodes = get_nodes_by_type(model_onnx, "FakeQuantize")

    inputs = [get_all_inputs_for_graph_node(fq_node, model_onnx.graph) for fq_node in fq_nodes]

    for quantizer, fq_parametres in zip(quantizers, inputs[1::2]):
        tensor_weight, _, __ = list(fq_parametres.values())
        # Quantize weights as they are exported quantized
        quantized_weights = quantizer.quantizer_module_ref(quantizer.quantized_module.weight).detach()

        diff = (quantized_weights.detach() - tensor_weight).abs()
        if (diff > 1e-6).any():
            assert ((diff[diff > 1e-6] - quant_len).abs() < 1e-6).all(), "quants completely different!"
            assert False, f"quant moved at flatten positions {torch.where(diff.flatten() > 1e-6)}"


def test_torch_onnx_export(tmp_path):
    model = TwoConvTestModel()
    nncf_config = get_config_for_export_mode(should_be_onnx_standard=False)

    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    onnx_checkpoint_path = tmp_path / "model.onnx"
    compression_ctrl.prepare_for_export()

    dummy_input = torch.randn(1, 1, 4, 4)
    torch.onnx.export(compression_model, dummy_input, onnx_checkpoint_path, verbose=False)
    onnx_model_proto = onnx.load_model(onnx_checkpoint_path)

    num_fq = 0
    num_model_nodes = 0
    num_other_nodes = 0

    for node in onnx_model_proto.graph.node:
        op_type = node.op_type
        if op_type == "FakeQuantize":
            num_fq += 1
        elif op_type in ["Conv", "Constant"]:
            num_model_nodes += 1
        else:
            num_other_nodes += 1
    assert num_fq == 4
    assert num_other_nodes == 0
