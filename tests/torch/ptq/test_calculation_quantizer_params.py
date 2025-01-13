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
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from torch import nn

from nncf import Dataset
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.min_max.torch_backend import PTMinMaxAlgoBackend
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.fake_quantize import calculate_quantizer_parameters
from nncf.quantization.fake_quantize import get_quantizer_narrow_range
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.torch.model_creation import wrap_model
from nncf.torch.statistics.aggregator import PTStatisticsAggregator
from tests.cross_fw.test_templates.test_calculate_quantizer_parameters import TemplateTestFQParams
from tests.torch.helpers import get_all_inputs_for_graph_node
from tests.torch.helpers import get_nodes_by_type

INPUT_SHAPE = (2, 3, 4, 5)


@dataclass
class CaseSymParams:
    fq_params: FakeQuantizeParameters
    per_channel: bool
    quant_group: QuantizerGroup
    ref_scale: np.ndarray


SYM_CASES = (
    CaseSymParams(
        fq_params=FakeQuantizeParameters(
            Tensor(torch.tensor(-0.49920455, dtype=torch.float32)),
            Tensor(torch.tensor(0.49530452, dtype=torch.float32)),
            Tensor(torch.tensor(-0.49920455, dtype=torch.float32)),
            Tensor(torch.tensor(0.49530452, dtype=torch.float32)),
            256,
        ),
        per_channel=False,
        quant_group=QuantizerGroup.ACTIVATIONS,
        ref_scale=0.49530452,
    ),
    CaseSymParams(
        fq_params=FakeQuantizeParameters(
            Tensor(torch.tensor(-0.49530452, dtype=torch.float32)),
            Tensor(torch.tensor(0.49530452, dtype=torch.float32)),
            Tensor(torch.tensor(-0.49530452, dtype=torch.float32)),
            Tensor(torch.tensor(0.49530452, dtype=torch.float32)),
            255,
        ),
        per_channel=False,
        quant_group=QuantizerGroup.WEIGHTS,
        ref_scale=0.49530452,
    ),
    CaseSymParams(
        fq_params=FakeQuantizeParameters(
            Tensor(torch.tensor([-0.4835594, -0.49530452, -0.49221927], dtype=torch.float32).reshape(1, 3, 1, 1)),
            Tensor(torch.tensor([0.4797816, 0.49920455, 0.48837382], dtype=torch.float32).reshape(1, 3, 1, 1)),
            Tensor(torch.tensor([-0.4835594, -0.49530452, -0.49221927], dtype=torch.float32).reshape(1, 3, 1, 1)),
            Tensor(torch.tensor([0.4797816, 0.49920455, 0.48837382], dtype=torch.float32).reshape(1, 3, 1, 1)),
            256,
        ),
        per_channel=True,
        quant_group=QuantizerGroup.ACTIVATIONS,
        ref_scale=torch.tensor([0.4797816, 0.49920455, 0.48837382]).reshape(1, 3, 1, 1),
    ),
    CaseSymParams(
        fq_params=FakeQuantizeParameters(
            Tensor(torch.tensor([-0.48837382, -0.49530452], dtype=torch.float32).reshape(2, 1, 1, 1)),
            Tensor(torch.tensor([0.48837382, 0.49530452], dtype=torch.float32).reshape(2, 1, 1, 1)),
            Tensor(torch.tensor([-0.48837382, -0.49530452], dtype=torch.float32).reshape(2, 1, 1, 1)),
            Tensor(torch.tensor([0.48837382, 0.49530452], dtype=torch.float32).reshape(2, 1, 1, 1)),
            255,
        ),
        per_channel=True,
        quant_group=QuantizerGroup.WEIGHTS,
        ref_scale=torch.tensor([0.48837382, 0.49530452]).reshape(2, 1, 1, 1),
    ),
)


@pytest.mark.parametrize("case_to_test", SYM_CASES)
def test_quantizer_params_sym(case_to_test: CaseSymParams):
    per_ch = case_to_test.per_channel
    fq_params = case_to_test.fq_params
    quant_group = case_to_test.quant_group
    qconfig = QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=per_ch)

    if not per_ch:
        scale_shape = [1]
    else:
        scale_shape = [1] * len(INPUT_SHAPE)
        axis = 0 if quant_group == QuantizerGroup.WEIGHTS else 1
        scale_shape[axis] = INPUT_SHAPE[axis]

    target_type = (
        TargetType.OPERATION_WITH_WEIGHTS if quant_group == QuantizerGroup.WEIGHTS else TargetType.PRE_LAYER_OPERATION
    )
    quantizer = PTMinMaxAlgoBackend._create_quantizer(qconfig, scale_shape, fq_params, target_type)

    assert quantizer.levels == fq_params.levels
    scale = quantizer.scale.detach().numpy()
    ref_scale = case_to_test.ref_scale
    assert np.allclose(scale, ref_scale)


@dataclass
class CaseAsymParams:
    fq_params: FakeQuantizeParameters
    per_channel: bool
    quant_group: QuantizerGroup
    ref_inp_low: np.ndarray
    ref_inp_range: np.ndarray


ASYM_CASES = (
    CaseAsymParams(
        fq_params=FakeQuantizeParameters(
            Tensor(torch.tensor(-0.49530452, dtype=torch.float32)),
            Tensor(torch.tensor(0.49143496, dtype=torch.float32)),
            Tensor(torch.tensor(-0.49530452, dtype=torch.float32)),
            Tensor(torch.tensor(0.49143496, dtype=torch.float32)),
            256,
        ),
        per_channel=False,
        quant_group=QuantizerGroup.WEIGHTS,
        ref_inp_low=-0.49530452,
        ref_inp_range=0.98673948,
    ),
    CaseAsymParams(
        fq_params=FakeQuantizeParameters(
            Tensor(torch.tensor(-0.49530452, dtype=torch.float32)),
            Tensor(torch.tensor(0.49143496, dtype=torch.float32)),
            Tensor(torch.tensor(-0.49530452, dtype=torch.float32)),
            Tensor(torch.tensor(0.49143496, dtype=torch.float32)),
            256,
        ),
        per_channel=False,
        quant_group=QuantizerGroup.ACTIVATIONS,
        ref_inp_low=-0.49530452,
        ref_inp_range=0.98673948,
    ),
    CaseAsymParams(
        fq_params=FakeQuantizeParameters(
            Tensor(torch.tensor([-0.48051512, -0.49776307, -0.44099426], dtype=torch.float32).reshape(1, 3, 1, 1)),
            Tensor(torch.tensor([0.4767611, 0.47861832, 0.48837382], dtype=torch.float32).reshape(1, 3, 1, 1)),
            Tensor(torch.tensor([-0.48051512, -0.49776307, -0.44099426], dtype=torch.float32).reshape(1, 3, 1, 1)),
            Tensor(torch.tensor([0.4767611, 0.47861832, 0.48837382], dtype=torch.float32).reshape(1, 3, 1, 1)),
            256,
        ),
        per_channel=True,
        quant_group=QuantizerGroup.ACTIVATIONS,
        ref_inp_low=torch.tensor([-0.48051512, -0.49776307, -0.44099426]).reshape(1, 3, 1, 1),
        ref_inp_range=torch.tensor([0.9572762, 0.9763814, 0.9293681]).reshape(1, 3, 1, 1),
    ),
    CaseAsymParams(
        fq_params=FakeQuantizeParameters(
            Tensor(torch.tensor([-0.4845584, -0.49583155], dtype=torch.float32).reshape(2, 1, 1, 1)),
            Tensor(torch.tensor([0.48837382, 0.4767611], dtype=torch.float32).reshape(2, 1, 1, 1)),
            Tensor(torch.tensor([-0.4845584, -0.49583155], dtype=torch.float32).reshape(2, 1, 1, 1)),
            Tensor(torch.tensor([0.48837382, 0.4767611], dtype=torch.float32).reshape(2, 1, 1, 1)),
            256,
        ),
        per_channel=True,
        quant_group=QuantizerGroup.WEIGHTS,
        ref_inp_low=torch.tensor([-0.4845584, -0.49583155]).reshape(2, 1, 1, 1),
        ref_inp_range=torch.tensor([0.97293222, 0.97259265]).reshape(2, 1, 1, 1),
    ),
)


@pytest.mark.parametrize("case_to_test", ASYM_CASES)
def test_quantizer_params_asym(case_to_test: CaseSymParams):
    per_ch = case_to_test.per_channel
    fq_params = case_to_test.fq_params
    quant_group = case_to_test.quant_group
    qconfig = QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=per_ch)

    if not per_ch:
        scale_shape = [1]
    else:
        scale_shape = [1] * len(INPUT_SHAPE)
        axis = 0 if quant_group == QuantizerGroup.WEIGHTS else 1
        scale_shape[axis] = INPUT_SHAPE[axis]

    target_type = (
        TargetType.OPERATION_WITH_WEIGHTS if quant_group == QuantizerGroup.WEIGHTS else TargetType.PRE_LAYER_OPERATION
    )
    quantizer = PTMinMaxAlgoBackend._create_quantizer(qconfig, scale_shape, fq_params, target_type)
    assert quantizer.levels == fq_params.levels
    assert fns.allclose(quantizer.input_low.data, case_to_test.ref_inp_low)
    assert fns.allclose(quantizer.input_range.data, case_to_test.ref_inp_range)


class LinearTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(3, 1, 1)
        self.bn2 = nn.BatchNorm2d(1)
        with torch.no_grad():
            self.conv1.weight.copy_(torch.rand_like(self.conv1.weight) - 0.5)
            self.conv2.weight.copy_(torch.rand_like(self.conv2.weight) - 0.5)

    def forward(self, x):
        # input_shape = [1, 3, 32, 32]
        relu = self.relu(self.conv1(x))
        bn1 = self.bn1(relu)
        avg_pool = self.avg_pool(bn1)
        x2 = self.relu(self.conv2(avg_pool))
        out = self.bn2(x2)
        return out, relu, bn1, avg_pool


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self._length = 1
        self._img_size = 32

    def __getitem__(self, idx):
        if idx >= self._length:
            raise StopIteration
        test_input_sample = torch.zeros([3, self._img_size, self._img_size])
        for i in range(0, self._img_size):
            for j in range(0, self._img_size):
                test_input_sample[0][i][j] = i * self._img_size - j
        test_input_sample[1] = test_input_sample[0] + 0.5
        test_input_sample[2] = test_input_sample[0] * 0.5
        return test_input_sample

    def __len__(self):
        return self._length


def calculate_statistics(data, mode, qgroup, half_range=False):
    data = data.detach().numpy()
    per_ch = qgroup == QuantizerGroup.WEIGHTS
    axes = (1, 2, 3) if per_ch else None
    min_values = np.amin(data, axes)
    if mode == QuantizationMode.SYMMETRIC:
        max_values = np.amax(np.abs(data), axes)
    else:
        max_values = np.amax(data, axes)

    statistics = MinMaxTensorStatistic(
        min_values=Tensor(torch.tensor(min_values)), max_values=Tensor(torch.tensor(max_values))
    )
    signedness_to_force = True if qgroup == QuantizerGroup.WEIGHTS else None
    qconfig = QuantizerConfig(num_bits=8, mode=mode, per_channel=per_ch, signedness_to_force=signedness_to_force)
    narrow_range = get_quantizer_narrow_range(qconfig, qgroup)
    fq_params = calculate_quantizer_parameters(statistics, qconfig, qgroup, narrow_range, half_range)
    return {"input_low": fq_params.input_low, "input_high": fq_params.input_high}


def calculate_fq_params(model, input_data):
    _, relu, bn1, avg_pool = model(input_data)
    conv1_stats = calculate_statistics(input_data, QuantizationMode.SYMMETRIC, QuantizerGroup.ACTIVATIONS)
    bn1_stats = calculate_statistics(bn1, QuantizationMode.SYMMETRIC, QuantizerGroup.ACTIVATIONS)
    conv2_stats = calculate_statistics(avg_pool, QuantizationMode.SYMMETRIC, QuantizerGroup.ACTIVATIONS)
    avg_pool_stats = calculate_statistics(relu, QuantizationMode.SYMMETRIC, QuantizerGroup.ACTIVATIONS)

    conv1_w = model.conv1.weight
    conv1_w_stats = calculate_statistics(conv1_w, QuantizationMode.SYMMETRIC, QuantizerGroup.WEIGHTS, True)
    conv2_w = model.conv2.weight
    conv2_w_stats = calculate_statistics(conv2_w, QuantizationMode.SYMMETRIC, QuantizerGroup.WEIGHTS)
    return {
        "//nncf_model_input_0|OUTPUT/FakeQuantize": conv1_stats,
        "/bn1/LinearTestModel/BatchNorm2d[bn1]/batch_norm_0|INPUT0/FakeQuantize": bn1_stats,
        "/avg_pool/LinearTestModel/AdaptiveAvgPool2d[avg_pool]/adaptive_avg_pool2d_0|INPUT0/FakeQuantize": (
            avg_pool_stats
        ),
        "/conv2/LinearTestModel/Conv2d[conv2]/conv2d_0|INPUT0/FakeQuantize": conv2_stats,
        "/conv1/LinearTestModel/Conv2d[conv1]/conv2d_0|INPUT1/FakeQuantize": conv1_w_stats,
        "/conv2/LinearTestModel/Conv2d[conv2]/conv2d_0|INPUT1/FakeQuantize": conv2_w_stats,
    }


def test_quantizer_parameters_export(tmp_path: Path, _seed):
    model = LinearTestModel()
    model.eval().cpu()

    data_loader = torch.utils.data.DataLoader(SyntheticDataset(), batch_size=1)
    input_data = next(iter(data_loader))
    fq_params = calculate_fq_params(model, input_data)

    dataset = Dataset(data_loader)
    min_max_algo = MinMaxQuantization(subset_size=1, preset=QuantizationPreset.PERFORMANCE, inplace_statistics=False)
    statistics_aggregator = PTStatisticsAggregator(dataset)

    nncf_network = wrap_model(model, torch.ones([1, 3, 32, 32]), True)
    statistic_points = min_max_algo.get_statistic_points(nncf_network, nncf_network.nncf.get_graph())
    statistics_aggregator.register_statistic_points(statistic_points)
    statistics_aggregator.collect_statistics(model, nncf_network.nncf.get_graph())
    torch_quantized_model = min_max_algo.apply(
        nncf_network, nncf_network.nncf.get_graph(), statistics_aggregator.statistic_points
    )

    path = str(tmp_path / "torch_ptq_model.onnx")
    torch.onnx.export(
        torch_quantized_model,
        input_data,
        path,
        export_params=True,
        opset_version=13,
        do_constant_folding=False,
    )

    onnx_model = onnx.load(path)
    fq_nodes = get_nodes_by_type(onnx_model, "FakeQuantize")
    inputs = [get_all_inputs_for_graph_node(fq_node, onnx_model.graph) for fq_node in fq_nodes]
    torch_ptq_params = {}
    for fq_node, fq_input in zip(fq_nodes, inputs):
        fq_input = list(fq_input.values())
        input_low, input_high = fq_input[-2].flatten(), fq_input[-1].flatten()
        torch_ptq_params[fq_node.name] = {"input_low": input_low, "input_high": input_high}

    for name, param in fq_params.items():
        assert name in torch_ptq_params
        assert fns.allclose(param["input_low"], torch_ptq_params[name]["input_low"])
        assert fns.allclose(param["input_high"], torch_ptq_params[name]["input_high"])


class TestFQParams(TemplateTestFQParams):
    def to_nncf_tensor(self, t):
        return Tensor(torch.tensor(t))
