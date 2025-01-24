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
import os
from copy import deepcopy
from pathlib import Path
from typing import List

import pytest

from examples.torch.common.model_loader import load_model
from nncf import NNCFConfig
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.common.quantization.structs import WeightQuantizerId
from nncf.torch import register_default_init_args
from nncf.torch.quantization.algo import QuantizationController
from tests.cross_fw.shared.paths import EXAMPLES_DIR
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.quantization_helpers import get_quantization_config_without_range_init
from tests.torch.quantization.test_hawq_precision_init import check_bitwidth_graph
from tests.torch.test_models.synthetic import AddTwoConv


class ManualConfigTestParamsBase:
    def __init__(self, name: str, bit_stats: List[List[str]]):
        self.name = name
        self.bit_stats = bit_stats

    def _get_config_path(self):
        raise NotImplementedError

    def create_nncf_config(self):
        config_path = self._get_config_path()
        return NNCFConfig.from_json(str(config_path))

    @staticmethod
    def create_model(model_name):
        return load_model(model_name, pretrained=False)


class ManualSampleConfigTestParams(ManualConfigTestParamsBase):
    def _get_config_path(self):
        return EXAMPLES_DIR.joinpath("torch", "classification", "configs", "mixed_precision") / self.name


class ManualTestConfigTestParams(ManualConfigTestParamsBase):
    def _get_config_path(self) -> Path:
        return TEST_ROOT / "torch" / "data" / "configs" / "hawq" / self.name


class BitwidthDistributionStatistics:
    def __init__(self, num_wq_per_bitwidth, num_aq_per_bitwidth):
        self.num_wq_per_bitwidth = num_wq_per_bitwidth
        self.num_aq_per_bitwidth = num_aq_per_bitwidth


MANUAL_CONFIG_TEST_PARAMS = [
    ManualSampleConfigTestParams(
        name="mobilenet_v2_imagenet_mixed_int_manual.json",
        bit_stats=BitwidthDistributionStatistics(
            num_wq_per_bitwidth={4: 26, 8: 27}, num_aq_per_bitwidth={4: 36, 8: 28}
        ),
    ),
    ManualSampleConfigTestParams(
        name="mobilenet_v2_imagenet_mixed_int_manual_staged.json",
        bit_stats=BitwidthDistributionStatistics(
            num_wq_per_bitwidth={4: 26, 8: 27}, num_aq_per_bitwidth={4: 35, 8: 29}
        ),
    ),
    ManualSampleConfigTestParams(
        name="resnet50_imagenet_mixed_int_manual.json",
        bit_stats=BitwidthDistributionStatistics(
            num_wq_per_bitwidth={8: 27, 4: 27}, num_aq_per_bitwidth={8: 29, 4: 42}
        ),
    ),
    ManualSampleConfigTestParams(
        name="resnet50_imagenet_mixed_int_manual_staged.json",
        bit_stats=BitwidthDistributionStatistics(
            num_wq_per_bitwidth={8: 27, 4: 27}, num_aq_per_bitwidth={8: 35, 4: 36}
        ),
    ),
    ManualSampleConfigTestParams(
        name="squeezenet1_1_imagenet_mixed_int_manual.json",
        bit_stats=BitwidthDistributionStatistics(
            num_wq_per_bitwidth={8: 13, 4: 13}, num_aq_per_bitwidth={8: 16, 4: 11}
        ),
    ),
    ManualSampleConfigTestParams(
        name="squeezenet1_1_imagenet_mixed_int_manual_staged.json",
        bit_stats=BitwidthDistributionStatistics(
            num_wq_per_bitwidth={8: 13, 4: 13}, num_aq_per_bitwidth={8: 16, 4: 11}
        ),
    ),
    ManualTestConfigTestParams(
        name="resnet18_cifar10_mixed_int_manual.json",
        bit_stats=BitwidthDistributionStatistics(
            num_wq_per_bitwidth={8: 12, 4: 1, 2: 8}, num_aq_per_bitwidth={8: 11, 4: 11}
        ),
    ),
]


@pytest.mark.parametrize(
    "manual_config_params", MANUAL_CONFIG_TEST_PARAMS, ids=[p.name for p in MANUAL_CONFIG_TEST_PARAMS]
)
def test_hawq_manual_configs(manual_config_params):
    # Tip: check and correct configs with hardcoded layer names (bitwidth_per_scope attribute)
    # in case you changed quantized NNCFGraph and this test failed
    # with error like `Could not find a quantization point at scope name...`
    config = manual_config_params.create_nncf_config()
    config = register_default_init_args(config, create_ones_mock_dataloader(config), criterion=None)
    model = manual_config_params.create_model(config["model"])

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    nncf_stats = compression_ctrl.statistics()

    expected = manual_config_params.bit_stats
    actual = nncf_stats.quantization

    assert expected.num_wq_per_bitwidth == actual.num_wq_per_bitwidth
    assert expected.num_aq_per_bitwidth == actual.num_aq_per_bitwidth


class ManualSingleConvTestParams:
    ACTIVATION_SCOPE = "/nncf_model_input_0|OUTPUT"

    def __init__(self, name: str):
        self.name = name
        self.nncf_config = get_quantization_config_without_range_init()
        self.expects_error = False
        self.model = BasicConvTestModel()

    def for_device(self, target_device):
        self.nncf_config["target_device"] = target_device
        return self

    def raises_error(self):
        self.expects_error = True
        return self

    def num_bits_for_activation(self, num_bits):
        self.nncf_config["compression"]["initializer"].update(
            {"precision": {"bitwidth_per_scope": [[num_bits, self.ACTIVATION_SCOPE]]}}
        )
        return self


MANUAL_SINGLE_CONV_TEST_PARAMS = [
    ManualSingleConvTestParams(name="manual_init_multiple_int8_qconfigs").for_device("CPU").num_bits_for_activation(8),
    ManualSingleConvTestParams(name="manual_init_int4_sym_int8_asym").for_device("NPU").num_bits_for_activation(4),
    ManualSingleConvTestParams(name="manual_init_trial").for_device("TRIAL").num_bits_for_activation(4),
    ManualSingleConvTestParams(name="incompatible_bitwidth")
    .for_device("NPU")
    .num_bits_for_activation(2)
    .raises_error(),
]


@pytest.mark.parametrize("params", MANUAL_SINGLE_CONV_TEST_PARAMS, ids=[p.name for p in MANUAL_SINGLE_CONV_TEST_PARAMS])
def test_manual_single_conv(params):
    config = params.nncf_config
    register_bn_adaptation_init_args(config)
    model = params.model

    if params.expects_error:
        with pytest.raises(ValueError):
            create_compressed_model_and_algo_for_test(model, config)
    else:
        model, ctrl = create_compressed_model_and_algo_for_test(model, config)
        path_to_dot = "{}.dot".format(params.name)
        graph_dir = os.path.join("quantized", "hawq")
        check_bitwidth_graph(ctrl, model, path_to_dot, graph_dir)


class PrecisionInitTestDesc:
    def __init__(self):
        self.model_creator = AddTwoConv
        config = get_quantization_config_without_range_init()
        config["compression"]["initializer"].update(
            {
                "precision": {
                    "bitwidth_per_scope": [
                        [2, "AddTwoConv/NNCFConv2d[conv1]/conv2d_0|WEIGHT"],
                        [4, "AddTwoConv/NNCFConv2d[conv2]/conv2d_0|WEIGHT"],
                    ]
                }
            }
        )
        config["target_device"] = "TRIAL"
        config["compression"]["activations"] = {"bits": 6}
        self.config = config
        self.ref_bits = [
            (WeightQuantizerId(target_node_name="AddTwoConv/NNCFConv2d[conv1]/conv2d_0"), 2),
            (WeightQuantizerId(target_node_name="AddTwoConv/NNCFConv2d[conv2]/conv2d_0"), 4),
            (NonWeightQuantizerId(target_node_name="AddTwoConv/NNCFConv2d[conv2]/conv2d_0"), 6),
            (NonWeightQuantizerId(target_node_name="AddTwoConv/NNCFConv2d[conv1]/conv2d_0"), 6),
            (NonWeightQuantizerId("/nncf_model_input_0"), 6),
        ]
        self.expected_stats = BitwidthDistributionStatistics(
            num_wq_per_bitwidth={4: 1, 2: 1}, num_aq_per_bitwidth={6: 3}
        )
        self.config_to_resume = None

    def __str__(self):
        return "resume_with_same" if self.config == self.config_to_resume else "resume_without_init"

    def config_with_all_inits(self):
        self.config["compression"]["initializer"].update(
            {"range": {"num_init_samples": 1}, "batchnorm_adaptation": {"num_bn_adaptation_samples": 1}}
        )
        return self

    def resume_with_the_same_config(self):
        self.config_to_resume = deepcopy(self.config)
        return self

    def resume_with_the_same_config_without_init(self):
        self.config_to_resume = deepcopy(self.config)
        self.config_to_resume["compression"]["initializer"] = {}
        return self

    @staticmethod
    def setup_init_spies(mocker):
        from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
        from nncf.torch.quantization.algo import QuantizationBuilder
        from nncf.torch.quantization.precision_init.manual_init import ManualPrecisionInitializer

        parse_range_init = mocker.spy(QuantizationBuilder, "_parse_range_init_params")
        get_stats = mocker.spy(QuantizationBuilder, "_get_statistics_for_final_range_init")
        run_bn_adapt = mocker.spy(BatchnormAdaptationAlgorithm, "run")
        apply_manual_precision_init = mocker.spy(ManualPrecisionInitializer, "apply_init")
        return [get_stats, parse_range_init, run_bn_adapt, apply_manual_precision_init]

    def check_precision_init(self, compression_ctrl: QuantizationController):
        for qid, quantizer in compression_ctrl.all_quantizations.items():
            expected_bit = [ref_bit for (ref_qid, ref_bit) in self.ref_bits if ref_qid == qid][0]
            assert quantizer.num_bits == expected_bit, "Unexpected number of bits for {}".format(str(qid))

        nncf_stats = compression_ctrl.statistics()
        actual_stats = nncf_stats.quantization

        assert self.expected_stats.num_wq_per_bitwidth == actual_stats.num_wq_per_bitwidth
        assert self.expected_stats.num_aq_per_bitwidth == actual_stats.num_aq_per_bitwidth


def test_quantization_configs__with_precisions_list():
    desc = PrecisionInitTestDesc()
    model = desc.model_creator()
    config = desc.config
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    desc.check_precision_init(compression_ctrl)
