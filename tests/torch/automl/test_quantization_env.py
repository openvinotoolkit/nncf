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
import math
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from nncf import NNCFConfig
from nncf.common.hardware.config import HWConfig
from nncf.common.hardware.config import HWConfigType
from nncf.common.quantization.quantizer_setup import MultiConfigQuantizerSetup
from nncf.torch.automl.environment.quantization_env import ModelSizeCalculator
from nncf.torch.automl.environment.quantization_env import QuantizationEnv
from nncf.torch.automl.environment.quantization_env import QuantizationEnvParams
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.hardware.config import PTHWConfig
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.algo import ExperimentalQuantizationBuilder
from nncf.torch.quantization.algo import PropagationBasedQuantizerSetupGenerator
from nncf.torch.quantization.precision_constraints import HardwareQuantizationConstraints
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import create_conv
from tests.torch.helpers import create_ones_mock_dataloader


def create_test_quantization_env(model_creator=BasicConvTestModel, input_info_cfg=None) -> QuantizationEnv:
    if input_info_cfg is None:
        input_info_cfg = {"input_info": {"sample_size": [1, 1, 4, 4]}}

    model = model_creator()
    nncf_network = NNCFNetwork(model, input_info=FillerInputInfo.from_nncf_config(input_info_cfg))
    hw_config_type = HWConfigType.NPU
    hw_config_path = HWConfig.get_path_to_hw_config(hw_config_type)
    hw_config = PTHWConfig.from_json(hw_config_path)
    setup = PropagationBasedQuantizerSetupGenerator(NNCFConfig(), nncf_network, hw_config=hw_config).generate_setup()
    dummy_multi_setup = MultiConfigQuantizerSetup.from_single_config_setup(setup)
    for qp in dummy_multi_setup.quantization_points.values():
        qconf_constraint_list = []
        qconf = qp.possible_qconfigs[0]
        bit_set = [8, 4, 2] if "conv" in str(qp.insertion_point) else [8, 4]
        for bits in bit_set:
            adj_qconf = deepcopy(qconf)
            adj_qconf.num_bits = bits
            qconf_constraint_list.append(adj_qconf)
        qp.possible_qconfigs = qconf_constraint_list
    experimental_builder = ExperimentalQuantizationBuilder(dummy_multi_setup, setup, {}, hw_config)
    experimental_builder.apply_to(nncf_network)
    experimental_ctrl = experimental_builder.build_controller(nncf_network)
    data_loader = create_ones_mock_dataloader(input_info_cfg)
    constraints = HardwareQuantizationConstraints()
    for qid, qp_id_set in experimental_ctrl.module_id_to_qp_id_translation_dict.items():
        first_qp_id_for_this_quantizer_module = next(iter(qp_id_set))
        qconfigs = dummy_multi_setup.quantization_points[first_qp_id_for_this_quantizer_module].possible_qconfigs
        constraints.add(qid, qconfigs)

    return QuantizationEnv(
        nncf_network,
        experimental_ctrl,
        constraints,
        data_loader,
        lambda *x: 0,
        hw_config_type=HWConfigType.NPU,
        params=QuantizationEnvParams(
            compression_ratio=0.15,
            eval_subset_ratio=1.0,
            skip_constraint=False,
            performant_bw=False,
            finetune=False,
            bits=[2, 4, 8],
            dump_init_precision_data=False,
        ),
    )


def test_can_create_quant_env():
    create_test_quantization_env()


QUANTIZER_CFG_TUPLES = [
    ([2, 2], [4, 2]),
    ([2, 4], [4, 4]),
    ([2, 8], [4, 4]),
    ([4, 4], [4, 4]),
    ([4, 8], [4, 4]),
    ([1, 6], [4, 4]),
    ([16, 1], [8, 2]),
]


@pytest.mark.parametrize(
    "bitwidth_cfg_tuple",
    QUANTIZER_CFG_TUPLES,
    ids=["_".join(["bitwidth_cfg", str(tup[0])]) for tup in QUANTIZER_CFG_TUPLES],
)
def test_step(bitwidth_cfg_tuple, mocker):
    final_cfg_spy = mocker.spy(ModelSizeCalculator, "__call__")
    bitwidth_cfg, final_cfg = bitwidth_cfg_tuple

    qenv = create_test_quantization_env()

    for i, bw in enumerate(bitwidth_cfg):
        observation, reward, is_done, info_set = qenv.step(bw)

        # no NaN in observation check
        assert not observation.isnull().values.any()
        assert reward == 0

        if i < len(bitwidth_cfg) - 1:
            assert is_done is False
            assert len(info_set) == 0
        else:
            assert is_done is True

            # Two factors impact final quantization policy
            # 1. Size constraint is enabled by default and targets 0.15 ratio
            # 2. Bitwidth space per quantizer.
            # Reference final policy is hardened as test truth.
            assert list(final_cfg_spy.call_args[0][1].values()) == final_cfg
            assert info_set["model_ratio"] == final_cfg[-1] / qenv.model_size_calculator.FLOAT_BITWIDTH


def test_overflow_step():
    qenv = create_test_quantization_env()

    for i, bw in enumerate([8, 8, 8]):
        if i < len(qenv.qctrl.all_quantizations):
            qenv.step(bw)
        else:
            # Extra stepping where step number is more than
            # the number of quantizers should throw IndexError
            with pytest.raises(IndexError):
                qenv.step(bw)


PRETRAINED_SCORE = [74.3, -0.11, None]
COMPRESSED_SCORE = [0.12, -7.23, "dummy"]
MODEL_SIZE_RATIO = [0.120, 2, False]


@pytest.mark.parametrize(
    "model_size_ratio", MODEL_SIZE_RATIO, ids=["_".join(["size_ratio", str(r)]) for r in MODEL_SIZE_RATIO]
)
@pytest.mark.parametrize(
    "compressed_score", COMPRESSED_SCORE, ids=["_".join(["compressed_score", str(s)]) for s in COMPRESSED_SCORE]
)
@pytest.mark.parametrize(
    "pretrained_score", PRETRAINED_SCORE, ids=["_".join(["pretrained_score", str(s)]) for s in PRETRAINED_SCORE]
)
def test_reward(pretrained_score, compressed_score, model_size_ratio):
    qenv = create_test_quantization_env()
    qenv.pretrained_score = pretrained_score

    if any(map(lambda x: not isinstance(x, (int, float)), [pretrained_score, compressed_score, model_size_ratio])):
        with pytest.raises(TypeError):
            qenv.reward(compressed_score, model_size_ratio)
    else:
        reward = qenv.reward(compressed_score, model_size_ratio)
        assert reward == compressed_score * (10 ** (-np.floor(math.log(abs(pretrained_score), 10))))


STRATEGY_LIST = [[2], [8, 8], [4, 8, 2]]

SKIP_CONSTRAINT_BOOL = [True, False]


@pytest.mark.parametrize(
    "skip_bool", SKIP_CONSTRAINT_BOOL, ids=["_".join(["skip_constraint", str(s)]) for s in SKIP_CONSTRAINT_BOOL]
)
@pytest.mark.parametrize(
    "strategy", STRATEGY_LIST, ids=["_".join(["bitwidth_strategy", str(s)]) for s in STRATEGY_LIST]
)
def test_evaluate_strategy(strategy, skip_bool, mocker):
    final_cfg_spy = mocker.spy(ModelSizeCalculator, "__call__")
    qenv = create_test_quantization_env()

    if len(strategy) != len(qenv.qctrl.all_quantizations):
        with pytest.raises(AssertionError):
            qenv.evaluate_strategy(strategy, skip_constraint=skip_bool)
    else:
        observation, reward, is_done, info_set = qenv.evaluate_strategy(strategy, skip_constraint=skip_bool)

        evaluated_strategy = list(final_cfg_spy.call_args[0][1].values())

        assert not observation.isnull().values.any()
        assert reward == 0
        assert is_done is True

        if skip_bool is True:
            assert info_set["model_ratio"] == strategy[-1] / qenv.model_size_calculator.FLOAT_BITWIDTH
        else:
            assert info_set["model_ratio"] == evaluated_strategy[-1] / qenv.model_size_calculator.FLOAT_BITWIDTH


def check_bw_cfg(qenv, input_strategy, output_ref):
    if len(input_strategy) != len(qenv.qctrl.all_quantizations):
        with pytest.raises(AssertionError):
            qenv.evaluate_strategy(input_strategy, skip_constraint=True)
        return
    _ = qenv.evaluate_strategy(input_strategy, skip_constraint=True)
    evaluated_strategy = qenv.master_df["action"].values.tolist()
    assert evaluated_strategy == output_ref


def check_both_bw_assignment_modes(qenv, strategy):
    input_strategy = strategy[0]

    # qenv.performant_bw is false by default
    output_ref = input_strategy
    check_bw_cfg(qenv, input_strategy, output_ref)

    # set qenv.performant_bw to True to enable bw_align_flow
    qenv.performant_bw = True
    output_ref = strategy[1]
    check_bw_cfg(qenv, input_strategy, output_ref)


STRATEGY_ONE_AQ_ONE_WQ_LIST = [
    [[8, 8], [8, 8]],
    [[8, 4], [4, 4]],
    [[8, 2], [4, 2]],
    [[4, 8], [4, 4]],
    [[4, 4], [4, 4]],
    [[4, 2], [4, 2]],
]


@pytest.mark.parametrize(
    "strategy",
    STRATEGY_ONE_AQ_ONE_WQ_LIST,
    ids=["_".join(["bitwidth_strategy", str(s[0])]) for s in STRATEGY_ONE_AQ_ONE_WQ_LIST],
)
def test_align_bw_action_one_aq_one_wq(strategy, mocker):
    qenv = create_test_quantization_env()
    check_both_bw_assignment_modes(qenv, strategy)


STRATEGY_ONE_AQ_TWO_WQ_LIST = [
    [[4, 2, 2], [4, 2, 2]],
    [[4, 2, 4], [4, 2, 4]],
    [[4, 2, 8], [4, 2, 4]],
    [[4, 4, 2], [4, 4, 2]],
    [[4, 4, 4], [4, 4, 4]],
    [[4, 4, 8], [4, 4, 4]],
    [[4, 8, 2], [4, 4, 2]],
    [[4, 8, 4], [4, 4, 4]],
    [[4, 8, 8], [4, 4, 4]],
    [[8, 2, 2], [4, 2, 2]],
    [[8, 2, 4], [4, 2, 4]],
    [[8, 2, 8], [4, 2, 4]],
    [[8, 4, 2], [4, 4, 2]],
    [[8, 4, 4], [4, 4, 4]],
    [[8, 4, 8], [4, 4, 4]],
    [[8, 8, 2], [4, 4, 2]],
    [[8, 8, 4], [4, 4, 4]],
    [[8, 8, 8], [8, 8, 8]],
]


@pytest.mark.parametrize(
    "strategy",
    STRATEGY_ONE_AQ_TWO_WQ_LIST,
    ids=["_".join(["bitwidth_strategy", str(s[0])]) for s in STRATEGY_ONE_AQ_TWO_WQ_LIST],
)
def test_align_bw_action_one_aq_two_wq(strategy, mocker):
    class OneAQTwoWQTestModel(BasicConvTestModel):
        def __init__(self):
            super().__init__()
            self.conv2 = create_conv(
                self.in_channels, self.out_channels, self.kernel_size, self.weight_init, self.bias_init
            )

        def forward(self, x):
            return (self.conv(x), self.conv2(x))

    qenv = create_test_quantization_env(model_creator=OneAQTwoWQTestModel)
    check_both_bw_assignment_modes(qenv, strategy)


STRATEGY_TWO_AQ_ONE_WQ_LIST = [
    [[4, 4, 2], [4, 4, 2]],
    [[4, 4, 4], [4, 4, 4]],
    [[4, 4, 8], [4, 4, 4]],
    [[4, 8, 2], [4, 4, 2]],
    [[4, 8, 4], [4, 4, 4]],
    [[4, 8, 8], [4, 4, 4]],
    [[8, 4, 2], [4, 4, 2]],
    [[8, 4, 4], [4, 4, 4]],
    [[8, 4, 8], [4, 4, 4]],
    [[8, 8, 2], [4, 4, 2]],
    [[8, 8, 4], [4, 4, 4]],
    [[8, 8, 8], [8, 8, 8]],
]


@pytest.mark.parametrize(
    "strategy",
    STRATEGY_TWO_AQ_ONE_WQ_LIST,
    ids=["_".join(["bitwidth_strategy", str(s[0])]) for s in STRATEGY_TWO_AQ_ONE_WQ_LIST],
)
def test_align_bw_action_two_aq_one_wq(strategy, mocker):
    class TwoAQOneWQTestModel(BasicConvTestModel):
        def __init__(self):
            super().__init__()
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)

        def forward(self, x1, x2):
            return self.conv(torch.cat([self.act1(x1), self.act2(x2)]))

    qenv = create_test_quantization_env(
        model_creator=TwoAQOneWQTestModel,
        input_info_cfg={"input_info": [{"sample_size": [1, 1, 4, 4]}, {"sample_size": [1, 1, 4, 4]}]},
    )

    check_both_bw_assignment_modes(qenv, strategy)


NPU_UMMAPPABLE_STRATEGY = [[2, 8], [2, 4], [2, 2]]


@pytest.mark.parametrize(
    "strategy",
    NPU_UMMAPPABLE_STRATEGY,
    ids=["_".join(["npu_unmappable_strategy", str(s)]) for s in NPU_UMMAPPABLE_STRATEGY],
)
def test_select_config_for_actions(strategy):
    qenv = create_test_quantization_env()

    with pytest.raises(AssertionError):
        qenv.select_config_for_actions(strategy)
