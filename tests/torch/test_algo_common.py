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
import copy
import logging
import os
from functools import reduce
from typing import Dict, List

import onnx
import pytest
import torch
from torch import cuda
from torch import nn

import nncf
from nncf import NNCFConfig
from nncf.api.compression import CompressionStage
from nncf.config.schemata.defaults import VALIDATE_SCOPES
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import DOMAIN_CUSTOM_OPS_NAME
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import PTTensorListComparator
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.pruning.helpers import get_basic_pruning_config
from tests.torch.quantization.quantization_helpers import get_quantization_config_without_range_init
from tests.torch.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config
from tests.torch.sparsity.rb.test_algo import get_basic_sparsity_config
from tests.torch.test_models.synthetic import ConvRelu6HSwishHSigmoid


class BasicLinearTestModel(nn.Module):
    def __init__(self, size=4):
        super().__init__()
        self.fc = nn.Linear(size, size)

    def forward(self, x):
        return self.fc(x)


class BasicTestModelWithTwoInputOutput(nn.Module):
    def __init__(self, size=4):
        super().__init__()
        self.fc0 = nn.Linear(size, size)
        self.fc1 = nn.Linear(size, size)

    def forward(self, x0, x1):
        return self.fc0(x0), self.fc1(x1)


def get_const_sparsity_config():
    config = get_empty_config()
    config["compression"] = {"algorithm": "const_sparsity"}
    return config


def get_basic_asym_quantization_config(model_size=4):
    config = get_quantization_config_without_range_init(model_size)
    config["compression"]["activations"] = {"mode": "asymmetric"}
    config["compression"]["initializer"]["range"] = {"num_init_samples": 0}
    return config


def get_filter_pruning_config():
    config = get_basic_pruning_config()
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["params"]["prune_first_conv"] = True
    return config


@pytest.mark.parametrize(
    "config_provider",
    (
        get_quantization_config_without_range_init,
        get_basic_asym_quantization_config,
        get_basic_sparsity_config,
        get_basic_magnitude_sparsity_config,
        get_const_sparsity_config,
        get_filter_pruning_config,
    ),
    ids=("SymQuantization", "AsymQuantization", "Sparsity", "MagnitudeSparsity", "ConstSparsity", "FilterPruning"),
)
@pytest.mark.parametrize("model_provider", (ConvRelu6HSwishHSigmoid, BasicLinearTestModel), ids=("Conv2d", "Linear"))
class TestCompressionAlgos:
    def test_can_export_compressed_model(self, tmp_path, config_provider, model_provider):
        test_path = str(tmp_path.joinpath("test.onnx"))
        model = model_provider()
        config = config_provider()
        register_bn_adaptation_init_args(config)
        compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

        state_before = copy.deepcopy(compressed_model.state_dict())
        compression_ctrl.export_model(test_path)
        state_after = compressed_model.state_dict()

        assert os.path.exists(test_path)
        PTTensorListComparator.check_equal(list(state_before.values()), list(state_after.values()))


@pytest.mark.parametrize(
    ("config_provider", "mask_getter"),
    [
        (get_basic_sparsity_config, lambda x: x.mask),
        (get_filter_pruning_config, lambda x: x.binary_filter_pruning_mask),
    ],
    ids=("sparsity", "filter_pruning"),
)
def test_no_weight_override_on_pruning_export(tmp_path, config_provider, mask_getter):
    test_path = str(tmp_path.joinpath("test.onnx"))
    model = ConvRelu6HSwishHSigmoid()
    config = config_provider()
    register_bn_adaptation_init_args(config)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    operand = compressed_model.conv1.get_pre_op("0").op
    with torch.no_grad():
        mask = mask_getter(operand)
        mask[mask != 0] = 0  # intentionally corrupt mask to zero out weights
    state_before = copy.deepcopy(compressed_model.state_dict())
    compression_ctrl.export_model(test_path)
    state_after = compressed_model.state_dict()

    PTTensorListComparator.check_equal(list(state_before.values()), list(state_after.values()))


class ConfigCreator:
    def __init__(self):
        self._config = get_empty_config()
        self._algorithm_sections = {}

    def create(self) -> NNCFConfig:
        self._config["compression"] = []
        for algo_name, params in self._algorithm_sections.items():
            algo_section = {"algorithm": algo_name}
            if params:
                algo_section["params"] = params
            self._config["compression"].append(algo_section)
        return copy.deepcopy(self._config)

    def add_algo(self, name: str, params: Dict = None):
        self._algorithm_sections[name] = params
        return self

    def __str__(self):
        return "_".join(self._algorithm_sections)


class CompressionStageTestStruct:
    def __init__(self, config_provider: "ConfigCreator", compression_stages: List[CompressionStage]):
        self.config_provider = config_provider
        self.compression_stages = compression_stages

    def __str__(self):
        return str(self.config_provider)


staged_quantization_params = {"activations_quant_start_epoch": 1, "weights_quant_start_epoch": 2}
magnitude_sparsity_params = {
    "schedule": "multistep",
    "multistep_steps": [1, 2],
    "multistep_sparsity_levels": [0, 0.3, 0.5],
}
filter_pruning_params = {"schedule": "exponential", "num_init_steps": 0, "pruning_steps": 3}
FFF_levels = [CompressionStage.FULLY_COMPRESSED] * 3
NPF_levels = [CompressionStage.UNCOMPRESSED, CompressionStage.PARTIALLY_COMPRESSED, CompressionStage.FULLY_COMPRESSED]
LIST_OF_TEST_PARAMS = [
    CompressionStageTestStruct(config_provider=ConfigCreator().add_algo("quantization"), compression_stages=FFF_levels),
    CompressionStageTestStruct(
        config_provider=ConfigCreator().add_algo("quantization", staged_quantization_params),
        compression_stages=NPF_levels,
    ),
    CompressionStageTestStruct(
        config_provider=ConfigCreator().add_algo("const_sparsity"), compression_stages=FFF_levels
    ),
    CompressionStageTestStruct(
        config_provider=ConfigCreator().add_algo("magnitude_sparsity", magnitude_sparsity_params),
        compression_stages=NPF_levels,
    ),
    CompressionStageTestStruct(
        config_provider=ConfigCreator().add_algo(
            "rb_sparsity",
            {
                "sparsity_target": 0.61,
                "sparsity_target_epoch": 2,
            },
        ),
        compression_stages=NPF_levels,
    ),
    CompressionStageTestStruct(
        config_provider=ConfigCreator().add_algo(
            "filter_pruning", {"num_init_steps": 1, "pruning_steps": 2, "schedule": "baseline"}
        ),
        compression_stages=[
            CompressionStage.UNCOMPRESSED,
            CompressionStage.FULLY_COMPRESSED,
            CompressionStage.FULLY_COMPRESSED,
        ],
    ),
    CompressionStageTestStruct(
        config_provider=ConfigCreator().add_algo("filter_pruning", filter_pruning_params), compression_stages=NPF_levels
    ),
    CompressionStageTestStruct(
        config_provider=ConfigCreator()
        .add_algo("magnitude_sparsity", magnitude_sparsity_params)
        .add_algo("quantization"),
        compression_stages=[CompressionStage.PARTIALLY_COMPRESSED] * 2 + [CompressionStage.FULLY_COMPRESSED],
    ),
    CompressionStageTestStruct(
        config_provider=ConfigCreator()
        .add_algo("magnitude_sparsity", magnitude_sparsity_params)
        .add_algo("quantization", staged_quantization_params),
        compression_stages=NPF_levels,
    ),
    CompressionStageTestStruct(
        config_provider=ConfigCreator()
        .add_algo("quantization", staged_quantization_params)
        .add_algo("filter_pruning", filter_pruning_params),
        compression_stages=NPF_levels,
    ),
    CompressionStageTestStruct(
        config_provider=ConfigCreator()
        .add_algo("magnitude_sparsity", magnitude_sparsity_params)
        .add_algo("quantization", staged_quantization_params)
        .add_algo("filter_pruning", filter_pruning_params),
        compression_stages=NPF_levels,
    ),
]


@pytest.mark.parametrize("test_struct", LIST_OF_TEST_PARAMS, ids=[str(param) for param in LIST_OF_TEST_PARAMS])
def test_can_get_compression_stage(test_struct: CompressionStageTestStruct):
    config_provider, compression_stages = test_struct.config_provider, test_struct.compression_stages
    model = BasicConvTestModel()
    config = config_provider.create()
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_scheduler = compression_ctrl.scheduler
    assert compression_ctrl.compression_stage() == compression_stages[0]

    compression_scheduler.epoch_step()
    assert compression_ctrl.compression_stage() == compression_stages[0]

    compression_scheduler.epoch_step()
    assert compression_ctrl.compression_stage() == compression_stages[1]

    compression_scheduler.epoch_step()
    assert compression_ctrl.compression_stage() == compression_stages[2]


@pytest.mark.parametrize(
    ("src", "dst", "ref"),
    (
        (CompressionStage.UNCOMPRESSED, CompressionStage.UNCOMPRESSED, CompressionStage.UNCOMPRESSED),
        (
            CompressionStage.PARTIALLY_COMPRESSED,
            CompressionStage.PARTIALLY_COMPRESSED,
            CompressionStage.PARTIALLY_COMPRESSED,
        ),
        (CompressionStage.FULLY_COMPRESSED, CompressionStage.FULLY_COMPRESSED, CompressionStage.FULLY_COMPRESSED),
        (CompressionStage.UNCOMPRESSED, CompressionStage.PARTIALLY_COMPRESSED, CompressionStage.PARTIALLY_COMPRESSED),
        (CompressionStage.UNCOMPRESSED, CompressionStage.FULLY_COMPRESSED, CompressionStage.PARTIALLY_COMPRESSED),
        (
            CompressionStage.PARTIALLY_COMPRESSED,
            CompressionStage.FULLY_COMPRESSED,
            CompressionStage.PARTIALLY_COMPRESSED,
        ),
    ),
)
def test_combo_of_compression_stages(src, dst, ref):
    assert src + dst == ref
    assert dst + src == ref
    src_c = copy.deepcopy(src)
    src_c += dst
    assert src_c == ref
    dst_c = copy.deepcopy(dst)
    dst_c += src
    assert dst_c == ref


def test_can_export_compressed_model_with_input_output_names(tmp_path):
    test_path = str(tmp_path.joinpath("test.onnx"))
    target_input_names = ["input1", "input2"]
    target_output_names = ["output1", "output2"]

    model = BasicTestModelWithTwoInputOutput()
    config = get_basic_asym_quantization_config()

    config["input_info"] = [{"sample_size": [1, 1, 4, 4]}, {"sample_size": [1, 1, 4, 4]}]
    register_bn_adaptation_init_args(config)

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    compression_ctrl.export_model(test_path, input_names=target_input_names, output_names=target_output_names)

    assert os.path.exists(test_path)

    onnx_model = onnx.load(test_path)

    curr_input_names = [node.name for node in onnx_model.graph.input]
    curr_output_names = [node.name for node in onnx_model.graph.output]

    assert curr_input_names == target_input_names
    assert curr_output_names == target_output_names


def test_can_export_compressed_model_with_specified_domain_for_custom_ops(tmp_path):
    test_path = str(tmp_path.joinpath("test.onnx"))

    model = BasicTestModelWithTwoInputOutput()
    config = get_basic_asym_quantization_config()

    config["input_info"] = [{"sample_size": [1, 1, 4, 4]}, {"sample_size": [1, 1, 4, 4]}]
    register_bn_adaptation_init_args(config)

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    compression_ctrl.export_model(test_path)

    assert os.path.exists(test_path)

    onnx_model = onnx.load(test_path)

    count_custom_ops = 0

    for op_node in onnx_model.graph.node:
        if op_node.op_type == "FakeQuantize":
            assert op_node.domain == DOMAIN_CUSTOM_OPS_NAME
            count_custom_ops += 1

    assert count_custom_ops == 4


def change_compression_algorithms_order(config):
    # changes order of compression algorithms in config
    def shift_list(list_for_shift):
        shifted_list = [list_for_shift.pop()] + list_for_shift
        return shifted_list

    config_compression = list(config.get("compression", {}))
    shifted_config_compression = shift_list(config_compression)
    config.update({"compression": shifted_config_compression})
    return config


def get_basic_rb_sparsity_int8_config():
    config = get_basic_sparsity_config()
    config.update(
        {
            "compression": [
                {
                    "algorithm": "rb_sparsity",
                    "sparsity_init": 0.02,
                    "params": {
                        "schedule": "polynomial",
                        "sparsity_target": 0.5,
                        "sparsity_target_epoch": 2,
                        "sparsity_freeze_epoch": 3,
                    },
                },
                {"algorithm": "quantization"},
            ]
        }
    )
    return config


comp_loss_configs = [
    get_basic_rb_sparsity_int8_config(),
    change_compression_algorithms_order(get_basic_rb_sparsity_int8_config()),
]


@pytest.mark.cuda
@pytest.mark.parametrize(
    "config",
    comp_loss_configs,
    ids=[
        reduce(lambda x, y: x + "_" + y.get("algorithm", ""), config.get("compression", []), "compression")
        for config in comp_loss_configs
    ],
)
@pytest.mark.skipif(not cuda.is_available(), reason="Since its GPU test, no need to run this without GPUs available")
def test_compression_loss_gpu_device_compatibility(config):
    model = BasicConvTestModel()
    model.to(cuda.current_device())
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_ctrl.loss()


NOT_SUPPORT_SCOPES_ALGO = ["knowledge_distillation", "NoCompressionAlgorithm", "elasticity", "progressive_shrinking"]


@pytest.mark.parametrize("algo_name", sorted(PT_COMPRESSION_ALGORITHMS.registry_dict.keys() - NOT_SUPPORT_SCOPES_ALGO))
@pytest.mark.parametrize("validate_scopes", (True, False, None))
def test_raise_validationerror_for_not_matched_scope_names(algo_name, validate_scopes):
    model = BasicLinearTestModel()
    config = ConfigCreator().add_algo(algo_name).create()
    config["compression"][0]["ignored_scopes"] = ["unknown"]

    if algo_name == "movement_sparsity":
        config["compression"][0]["params"] = {
            "warmup_start_epoch": 1,
            "warmup_end_epoch": 3,
            "enable_structured_masking": False,
            "init_importance_threshold": -0.1,
            "final_importance_threshold": 0.0,
            "importance_regularization_factor": 0.2,
            "power": 3,
            "steps_per_epoch": 4,
        }

    if validate_scopes is not None:
        config["compression"][0]["validate_scopes"] = validate_scopes

    if validate_scopes or (validate_scopes is None and VALIDATE_SCOPES is True):
        with pytest.raises(nncf.ValidationError, match="scope definitions"):
            create_compressed_model_and_algo_for_test(model, config)
    else:
        create_compressed_model_and_algo_for_test(model, config)


@pytest.mark.parametrize(
    "algos",
    (
        [
            "quantization",
        ],
        ["magnitude_sparsity", "filter_pruning"],
    ),
)
def test_compressed_model_has_controller_references(algos: List[str]):
    model = BasicLinearTestModel()
    cc = ConfigCreator()
    for algo_name in algos:
        cc.add_algo(algo_name)
    config = cc.create()
    register_bn_adaptation_init_args(config)
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert compression_ctrl is model.nncf.compression_controller


ALGOS_SUPPORTING_SINGLE_LINE_CONFIGS = [
    x
    for x in sorted(PT_COMPRESSION_ALGORITHMS.registry_dict.keys())
    if x
    not in [
        "knowledge_distillation",
        "movement_sparsity",
        "elasticity",
        "progressive_shrinking",
        "NoCompressionAlgorithm",
    ]
]


@pytest.mark.parametrize("algo_name", ALGOS_SUPPORTING_SINGLE_LINE_CONFIGS)
def test_can_apply_algo_with_single_line(algo_name, nncf_caplog):
    model = BasicLinearTestModel()
    config = ConfigCreator().add_algo(algo_name).create()
    with nncf_caplog.at_level(logging.INFO):
        create_compressed_model_and_algo_for_test(model, config)

    if algo_name == "quantization":
        assert "will not perform batchnorm adaptation" in nncf_caplog.text
