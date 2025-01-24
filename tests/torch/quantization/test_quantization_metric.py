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

from collections import namedtuple

import pytest

from nncf import NNCFConfig
from nncf.common.graph.patterns.manager import TargetDevice
from nncf.torch import create_compressed_model
from nncf.torch.quantization.metrics import MemoryConsumptionStatisticsCollector
from nncf.torch.quantization.metrics import ShareEdgesQuantizedDataPathStatisticsCollector
from tests.torch import test_models
from tests.torch.helpers import register_bn_adaptation_init_args


def get_basic_quantization_config():
    config = NNCFConfig()
    config.update(
        {
            "model": "AlexNet",
            "input_info": {
                "sample_size": [1, 3, 32, 32],
            },
            "compression": {
                "algorithm": "quantization",
                "quantize_inputs": True,
                "initializer": {"range": {"num_init_samples": 0}},
            },
        }
    )
    register_bn_adaptation_init_args(config)

    return config


def as_dict(obj):
    if isinstance(obj, list):
        return [as_dict(value) for value in obj]
    if isinstance(obj, dict):
        return {key: as_dict(value) for key, value in obj.items()}
    if hasattr(obj, "__dict__"):
        return {key: as_dict(value) for key, value in obj.__dict__.items() if not key.startswith("_")}
    return obj


CaseStruct = namedtuple(
    "CaseStruct", ("initializers", "activations", "weights", "ignored_scopes", "target_device", "expected")
)


QUANTIZATION_SHARE_AND_BITWIDTH_DISTR_STATS_TEST_CASES = [
    CaseStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=[],
        target_device="TRIAL",
        expected={
            "wq_counter": {
                "num_symmetric": 8,
                "num_asymmetric": 0,
                "num_signed": 0,
                "num_unsigned": 8,
                "num_per_tensor": 8,
                "num_per_channel": 0,
                "total_count": 8,
                "potential_count": 8,
            },
            "aq_counter": {
                "num_symmetric": 8,
                "num_asymmetric": 0,
                "num_signed": 0,
                "num_unsigned": 8,
                "num_per_tensor": 8,
                "num_per_channel": 0,
                "total_count": 8,
                "potential_count": None,
            },
            "num_wq_per_bitwidth": {8: 8},
            "num_aq_per_bitwidth": {8: 8},
        },
    ),
    CaseStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=[],
        target_device="CPU",
        expected={
            "wq_counter": {
                "num_symmetric": 8,
                "num_asymmetric": 0,
                "num_signed": 8,
                "num_unsigned": 0,
                "num_per_tensor": 0,
                "num_per_channel": 8,
                "total_count": 8,
                "potential_count": 8,
            },
            "aq_counter": {
                "num_symmetric": 8,
                "num_asymmetric": 0,
                "num_signed": 0,
                "num_unsigned": 8,
                "num_per_tensor": 8,
                "num_per_channel": 0,
                "total_count": 8,
                "potential_count": None,
            },
            "num_wq_per_bitwidth": {8: 8},
            "num_aq_per_bitwidth": {8: 8},
        },
    ),
]


@pytest.mark.parametrize("data", QUANTIZATION_SHARE_AND_BITWIDTH_DISTR_STATS_TEST_CASES)
def test_quantization_share_and_bitwidth_distribution_stats(data):
    config = get_basic_quantization_config()
    config["compression"]["initializer"].update(data.initializers)
    config["compression"]["activations"] = data.activations
    config["compression"]["weights"] = data.weights
    config["compression"]["ignored_scopes"] = data.ignored_scopes
    config["target_device"] = data.target_device

    ctrl, _ = create_compressed_model(test_models.AlexNet(), config)
    nncf_stats = ctrl.statistics()
    quantization_stats = nncf_stats.quantization

    for attr_name, expected_value in data.expected.items():
        actual_value = as_dict(getattr(quantization_stats, attr_name))
        assert expected_value == actual_value


MEMORY_CONSUMPTION_STATS_TEST_CASES = [
    CaseStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=[],
        target_device="TRIAL",
        expected={
            "fp32_weight_size": 88.74,
            "quantized_weight_size": 22.18,
            "max_fp32_activation_size": 0.0625,
            "max_compressed_activation_size": 0.015625,
            "weight_memory_consumption_decrease": 4.0,
        },
    ),
    CaseStruct(
        initializers={
            "precision": {
                "bitwidth_per_scope": [
                    [2, "AlexNet/Sequential[features]/NNCFConv2d[0]/conv2d_0|WEIGHT"],
                    [4, "AlexNet/Sequential[features]/NNCFConv2d[6]/conv2d_0|WEIGHT"],
                ]
            }
        },
        activations={},
        weights={"bits": 8},
        ignored_scopes=[],
        target_device="TRIAL",
        expected={
            "fp32_weight_size": 88.74,
            "quantized_weight_size": 21.86,
            "max_fp32_activation_size": 0.0625,
            "max_compressed_activation_size": 0.015625,
            "weight_memory_consumption_decrease": 4.05,
        },
    ),
    CaseStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=["AlexNet/Sequential[features]/NNCFConv2d[0]/conv2d_0"],
        target_device="TRIAL",
        expected={
            "fp32_weight_size": 88.74,
            "quantized_weight_size": 22.19,
            "max_fp32_activation_size": 0.0625,
            "max_compressed_activation_size": 0.0625,
            "weight_memory_consumption_decrease": 3.99,
        },
    ),
]


@pytest.mark.parametrize("data", MEMORY_CONSUMPTION_STATS_TEST_CASES)
def test_memory_consumption_stats(data):
    config = get_basic_quantization_config()
    config["compression"]["initializer"].update(data.initializers)
    config["compression"]["weights"] = data.weights
    config["compression"]["ignored_scopes"] = data.ignored_scopes
    config["target_device"] = data.target_device

    ctrl, _ = create_compressed_model(test_models.AlexNet(), config)
    stats = MemoryConsumptionStatisticsCollector(
        ctrl.model, ctrl.weight_quantizers, ctrl.non_weight_quantizers
    ).collect()

    for attr_name, expected_value in data.expected.items():
        actual_value = getattr(stats, attr_name)
        assert expected_value == pytest.approx(actual_value, rel=1e-2)


QUANTIZATION_CONFIGURATION_STATS_TEST_CASES = [
    CaseStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=[],
        target_device="TRIAL",
        expected={"quantized_edges_in_cfg": 173, "total_edges_in_cfg": 177},
    ),
    CaseStruct(
        initializers={},
        activations={},
        weights={},
        ignored_scopes=[
            "Inception3/__add___0",
            "Inception3/__add___1",
            "Inception3/__add___2",
            "Inception3/__mul___0",
            "Inception3/__mul___1",
            "Inception3/__mul___2",
        ],
        target_device="TRIAL",
        expected={"quantized_edges_in_cfg": 173, "total_edges_in_cfg": 177},
    ),
]


@pytest.mark.parametrize("data", QUANTIZATION_CONFIGURATION_STATS_TEST_CASES)
def test_quantization_configuration_stats(data):
    config = get_basic_quantization_config()
    config["compression"]["ignored_scopes"] = data.ignored_scopes
    config["input_info"]["sample_size"] = [2, 3, 299, 299]

    ctrl, _ = create_compressed_model(test_models.Inception3(aux_logits=True, transform_input=True), config)
    stats = ShareEdgesQuantizedDataPathStatisticsCollector(ctrl.model, ctrl, TargetDevice.ANY).collect()

    for attr_name, expected_value in data.expected.items():
        actual_value = as_dict(getattr(stats, attr_name))
        assert expected_value == actual_value


def test_full_ignored_scope():
    config = get_basic_quantization_config()
    config["compression"]["ignored_scopes"] = ["{re}.*"]
    ctrl, _ = create_compressed_model(test_models.AlexNet(), config)
    ctrl.statistics()
