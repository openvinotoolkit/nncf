"""
 Copyright (c) 2023 Intel Corporation
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

from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple

import pytest

from nncf import NNCFConfig
from nncf.experimental.torch.nas.bootstrapNAS.search.search import DataLoaderType
from nncf.config.structures import BNAdaptationInitArgs
from nncf.experimental.torch.nas.bootstrapNAS import SearchAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import get_empty_config
from tests.torch.nas.creators import NAS_MODEL_DESCS
from tests.torch.nas.creators import create_bnas_model_and_ctrl_by_test_desc
from tests.torch.nas.creators import create_bootstrap_training_model_and_ctrl
from tests.torch.nas.models.synthetic import ThreeConvModel
from tests.torch.nas.test_all_elasticity import fixture_nas_model_name  # pylint: disable=unused-import


class SearchTestDesc(NamedTuple):
    model_creator: Any
    blocks_to_skip: List[List[str]] = None
    input_sizes: List[int] = [1, 3, 32, 32]
    algo_params: Dict = {}
    name: str = None
    mode: str = "auto"

    def __str__(self):
        if hasattr(self.model_creator, '__name__'):
            name = self.model_creator.__name__
        elif self.name is not None:
            name = self.name
        else:
            name = 'NOT_DEFINED'
        return name


def prepare_test_model(search_desc):
    model, ctrl = create_bnas_model_and_ctrl_by_test_desc(search_desc)
    elasticity_ctrl = ctrl.elasticity_controller
    config = {
        "input_info": {"sample_size": search_desc.input_sizes},
        "bootstrapNAS": {
            "training": {
                "elasticity": {
                    "available_elasticity_dims": ["depth", "width"]
                }
            },
            "search": {
                "algorithm": "NSGA2",
                "num_evals": 2,
                "population": 1,
                "batchnorm_adaptation": {
                    "num_bn_adaptation_samples": 2
                },
            }
        }
    }
    nncf_config = NNCFConfig.from_dict(config)
    bn_adapt_args = BNAdaptationInitArgs(data_loader=create_ones_mock_dataloader(nncf_config))
    nncf_config.register_extra_structs([bn_adapt_args])
    return model, elasticity_ctrl, nncf_config


def prepare_search_algorithm(nas_model_name: str):
    if 'inception_v3' in nas_model_name:
        pytest.skip(
            f'Skip test for {nas_model_name} as it fails because of 2 issues: '
            'not able to set DynamicInputOp to train-only layers (ticket 60976) and '
            'invalid padding update in elastic kernel (ticket 60990)')

    elif nas_model_name in ['efficient_net_b0', 'shufflenetv2']:
        pytest.skip(
            f'Skip test for {nas_model_name} as exploration is underway to better manage its search space'
        )
    model = NAS_MODEL_DESCS[nas_model_name][0]()
    nncf_config = get_empty_config(input_sample_sizes=NAS_MODEL_DESCS[nas_model_name][1])
    nncf_config['bootstrapNAS'] = {'training': {'algorithm': 'progressive_shrinking'}}
    nncf_config['input_info'][0].update({'filler': 'random'})
    nncf_config['bootstrapNAS']['search'] = {"algorithm": "NSGA2", "num_evals": 2, "population": 1}
    nncf_config = NNCFConfig.from_dict(nncf_config)
    model, ctrl = create_bootstrap_training_model_and_ctrl(model, nncf_config)
    elasticity_ctrl = ctrl.elasticity_controller
    elasticity_ctrl.multi_elasticity_handler.enable_all()
    return SearchAlgorithm.from_config(model, elasticity_ctrl, nncf_config)


def update_search_bn_adapt_section(nncf_config, bn_adapt_section_is_called):
    if not bn_adapt_section_is_called:
        nncf_config['bootstrapNAS']['search']['batchnorm_adaptation']['num_bn_adaptation_samples'] = 0


NAS_MODELS_SEARCH_ENCODING = {
    'resnet18': [1, 3, 7, 15, 1, 1, 3, 3, 7, 7, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 15],
    'resnet50': [7, 15, 31, 63, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7,
                 7, 7, 7, 7, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
    'densenet_121': [1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
                     0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 7, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
                     0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0,
                     3, 0, 3, 0, 3, 0, 15, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0,
                     3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'mobilenet_v2': [0, 0, 1, 2, 4, 0, 0, 2, 3, 3, 5, 5, 5, 11, 11, 11, 11, 17, 17, 17, 29, 29, 29, 9,
                     39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31],
    'vgg11': [1, 3, 7, 7, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'vgg11_k7': [1, 3, 7, 7, 15, 15, 15, 15, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    'unet': [1, 3, 7, 15, 31, 15, 7, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0],
    'squeezenet1_0': [2, 0, 1, 1, 0, 1, 1, 0, 3, 3, 0, 3, 3, 0, 5, 5, 0, 5, 5, 1, 7, 7, 1, 7, 7, 2, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
    'resnext29_32x4d': [7, 15, 31, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
    'pnasnetb': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 1, 0, 2, 0, 1, 0,
                 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1,
                 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0,
                 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 7],
    'ssd_mobilenet': [0, 1, 3, 3, 7, 7, 15, 15, 15, 15, 15, 15, 31, 31, 7, 15, 3, 7, 3, 7, 1, 3, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
    'ssd_vgg': [1, 1, 3, 3, 7, 7, 7, 15, 15, 15, 15, 15, 31, 31, 7, 15, 3, 7, 3, 7, 3, 7, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 7],
    'mobilenet_v3_small': [0, 0, 2, 6, 6, 0, 2, 3, 0, 8, 17, 17, 2, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 3, 3,
                           17, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                           0, 0, 0, 255],
    'tcn': [3]
}  # fmt:skip


class TestSearchAlgorithm:

    def test_activate_maximum_subnet_at_init(self):
        search_desc = SearchTestDesc(model_creator=ThreeConvModel,
                                     algo_params={'width': {'min_width': 1, 'width_step': 1}},
                                     input_sizes=ThreeConvModel.INPUT_SIZE,
                                     )
        model, elasticity_ctrl, nncf_config = prepare_test_model(search_desc)
        elasticity_ctrl.multi_elasticity_handler.enable_elasticity(ElasticityDim.WIDTH)
        SearchAlgorithm(model, elasticity_ctrl, nncf_config)
        config_init = elasticity_ctrl.multi_elasticity_handler.get_active_config()
        elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        assert config_init == elasticity_ctrl.multi_elasticity_handler.get_active_config()

    def test_design_upper_bounds(self, nas_model_name):
        search = prepare_search_algorithm(nas_model_name)
        assert search.vars_upper == NAS_MODELS_SEARCH_ENCODING[nas_model_name]
        assert search.num_vars == len(NAS_MODELS_SEARCH_ENCODING[nas_model_name])

    @pytest.mark.parametrize("bn_adapt_section_is_called", [False,True],
                              ids=["section_with_zero_num_samples", "section_with_non_zero_num_samples"])
    def test_bn_adapt(self, mocker, bn_adapt_section_is_called, tmp_path):
        search_desc = SearchTestDesc(model_creator=ThreeConvModel,
                                     algo_params={'width': {'min_width': 1, 'width_step': 1}},
                                     input_sizes=ThreeConvModel.INPUT_SIZE,
                                     )
        nncf_network, ctrl, nncf_config = prepare_test_model(search_desc)
        update_search_bn_adapt_section(nncf_config, bn_adapt_section_is_called)
        bn_adapt_run_patch = mocker.patch(
            "nncf.common.initialization.batchnorm_adaptation.BatchnormAdaptationAlgorithm.run")
        ctrl.multi_elasticity_handler.enable_all()
        search_algo = SearchAlgorithm(nncf_network, ctrl, nncf_config)

        def fake_acc_eval(*unused):
            return 0

        search_algo.run(fake_acc_eval, mocker.MagicMock(spec=DataLoaderType), tmp_path)
        if bn_adapt_section_is_called:
            bn_adapt_run_patch.assert_called()
        else:
            bn_adapt_run_patch.assert_not_called()


class TestSearchEvaluators:
    def test_create_default_evaluators(self, nas_model_name, tmp_path):
        if nas_model_name in ['squeezenet1_0', 'pnasnetb']:
            pytest.skip(
                f'Skip test for {nas_model_name} as it fails.')
        search = prepare_search_algorithm(nas_model_name)
        search.run(lambda model, val_loader: 0, None, tmp_path)
        evaluators = search.evaluator_handlers
        assert len(evaluators) == 2
        assert evaluators[0].name == 'MACs'
        assert evaluators[1].name == 'top1_acc'
