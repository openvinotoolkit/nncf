"""
 Copyright (c) 2021 Intel Corporation
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
from copy import deepcopy

import pytest

from nncf import NNCFConfig
from nncf.automl.environment.quantization_env import QuantizationEnv, ModelSizeCalculator, QuantizationEnvParams
from nncf.dynamic_graph.graph_builder import ModelInputInfo
from nncf.hw_config import HWConfigType, HWConfig
from nncf.nncf_network import NNCFNetwork
from nncf.quantization.algo import ExperimentalQuantizationBuilder, PropagationBasedQuantizerSetupGenerator
from nncf.quantization.precision_constraints import HardwareQuantizationConstraints
from tests.helpers import create_mock_dataloader, BasicConvTestModel


def create_test_quantization_env() -> QuantizationEnv:
    model = BasicConvTestModel()
    nncf_network = NNCFNetwork(model, input_infos=[ModelInputInfo([1, 1, 4, 4])])
    hw_config_type = HWConfigType.VPU
    hw_config_path = HWConfig.get_path_to_hw_config(hw_config_type)
    hw_config = HWConfig.from_json(hw_config_path)
    setup = PropagationBasedQuantizerSetupGenerator(NNCFConfig(),
                                                    nncf_network,
                                                    hw_config=hw_config).generate_setup()
    experimental_builder = ExperimentalQuantizationBuilder(setup, {}, hw_config)
    experimental_builder.apply_to(nncf_network)
    # pylint:disable=line-too-long
    experimental_ctrl = experimental_builder.build_controller(nncf_network)
    data_loader = create_mock_dataloader(
            {
                "sample_size": [1, 1, 4, 4],
            })
    constraints = HardwareQuantizationConstraints()
    for qid in experimental_ctrl.all_quantizations:
        qconf_constraint_list = []
        qconf = experimental_ctrl.all_quantizations[qid].get_quantizer_config()
        bit_set = [8, 4, 2] if 'conv' in str(qid) else [8, 4]
        for bits in bit_set:
            adj_qconf = deepcopy(qconf)
            adj_qconf.num_bits = bits
            qconf_constraint_list.append(adj_qconf)
        constraints.add(qid, qconf_constraint_list)

    return QuantizationEnv(nncf_network,
                           experimental_ctrl,
                           constraints,
                           data_loader,
                           lambda *x: 0,
                           hw_config_type=HWConfigType.VPU,
                           params=QuantizationEnvParams(compression_ratio=0.15,
                                                        eval_subset_ratio=1.0,
                                                        skip_constraint=False,
                                                        finetune=False,
                                                        bits=[2, 4, 8],
                                                        dump_init_precision_data=False))

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
@pytest.mark.parametrize('bitwidth_cfg_tuple', QUANTIZER_CFG_TUPLES,
                         ids=['_'.join(['bitwidth_cfg', str(tup[0])]) for tup in QUANTIZER_CFG_TUPLES])
def test_step(bitwidth_cfg_tuple, mocker):
    final_cfg_spy = mocker.spy(ModelSizeCalculator, "__call__")
    bitwidth_cfg, final_cfg = bitwidth_cfg_tuple

    qenv = create_test_quantization_env()

    for i, bw in enumerate(bitwidth_cfg):
        observation, reward, is_done, info_set = qenv.step(bw)

        # no NaN in observation check
        assert not observation.isnull().values.any()
        assert reward == 0

        if i < len(bitwidth_cfg)-1:
            assert is_done is False
            assert len(info_set) == 0
        else:
            assert is_done is True

            # Two factors impact final quantization policy
            # 1. Size constraint is enabled by default and targets 0.15 ratio
            # 2. Bitwidth space per quantizer.
            # Reference final policy is hardened as test truth.
            assert list(final_cfg_spy.call_args[0][1].values()) == final_cfg
            assert info_set['model_ratio'] == final_cfg[-1]/qenv.model_size_calculator.FLOAT_BITWIDTH


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

@pytest.mark.parametrize('model_size_ratio', MODEL_SIZE_RATIO,
                         ids=['_'.join(['size_ratio', str(r)]) for r in MODEL_SIZE_RATIO])
@pytest.mark.parametrize('compressed_score', COMPRESSED_SCORE,
                         ids=['_'.join(['compressed_score', str(s)]) for s in COMPRESSED_SCORE])
@pytest.mark.parametrize('pretrained_score', PRETRAINED_SCORE,
                         ids=['_'.join(['pretrained_score', str(s)]) for s in PRETRAINED_SCORE])
def test_reward(pretrained_score, compressed_score, model_size_ratio):
    qenv = create_test_quantization_env()
    qenv.pretrained_score = pretrained_score

    if any(map(lambda x: not isinstance(x, (int, float)),
               [pretrained_score, compressed_score, model_size_ratio])):
        with pytest.raises(TypeError):
            qenv.reward(compressed_score, model_size_ratio)
    else:
        reward = qenv.reward(compressed_score, model_size_ratio)
        assert reward == (compressed_score - pretrained_score) * 0.1


STRATEGY_LIST = [
    [2],
    [8, 8],
    [4, 8, 2]
]

SKIP_CONSTRAINT_BOOL = [True, False]

@pytest.mark.parametrize('skip_bool', SKIP_CONSTRAINT_BOOL,
                         ids=['_'.join(['skip_constraint', str(s)]) for s in SKIP_CONSTRAINT_BOOL])
@pytest.mark.parametrize('strategy', STRATEGY_LIST,
                         ids=['_'.join(['bitwidth_strategy', str(s)]) for s in STRATEGY_LIST])
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
            assert info_set['model_ratio'] == strategy[-1]/qenv.model_size_calculator.FLOAT_BITWIDTH
        else:
            assert info_set['model_ratio'] == evaluated_strategy[-1]/qenv.model_size_calculator.FLOAT_BITWIDTH
