"""
 Copyright (c) 2022 Intel Corporation
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
from collections import OrderedDict

from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple

import pytest

from nncf import NNCFConfig
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.config.extractors import get_bn_adapt_algo_kwargs
from nncf.config.structures import BNAdaptationInitArgs
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_depth import ElasticDepthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.training.progressive_shrinking_builder import ProgressiveShrinkingBuilder
from nncf.experimental.torch.nas.bootstrapNAS.training.progressive_shrinking_controller import \
    ProgressiveShrinkingController
from nncf.experimental.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor
from nncf.torch.nncf_network import NNCFNetwork
from tests.torch.helpers import create_ones_mock_dataloader, MockModel
from tests.torch.nas.creators import create_bnas_model_and_ctrl_by_test_desc
from tests.torch.nas.models.synthetic import ThreeConvModel
from tests.torch.nas.test_scheduler import fixture_schedule_params


class PSControllerTestDesc(NamedTuple):
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


def prepare_test_model(ps_ctrl_desc):
    model, ctrl = create_bnas_model_and_ctrl_by_test_desc(ps_ctrl_desc)
    elasticity_ctrl = ctrl.elasticity_controller
    config = {
        "input_info": {"sample_size": ps_ctrl_desc.input_sizes},
        "bootstrapNAS": {
            "training": {
                "batchnorm_adaptation": {
                    "num_bn_adaptation_samples": 2
                },
                "elasticity": {
                    "available_elasticity_dims": ["depth", "width"]
                }
            },
        }
    }
    nncf_config = NNCFConfig.from_dict(config)
    bn_adapt_args = BNAdaptationInitArgs(data_loader=create_ones_mock_dataloader(nncf_config))
    nncf_config.register_extra_structs([bn_adapt_args])
    return model, elasticity_ctrl, nncf_config


def update_train_bn_adapt_section(nncf_config, bn_adapt_section_is_called):
    if not bn_adapt_section_is_called:
        nncf_config['bootstrapNAS']['training']['batchnorm_adaptation']['num_bn_adaptation_samples'] = 0


# pylint: disable=protected-access
class TestProgressiveTrainingController:
    @pytest.mark.parametrize(('bn_adapt_section', 'is_called'), (("section_with_zero_num_samples", False),
                                                                 ("section_with_non_zero_num_samples", True)))
    def test_bn_adapt(self, mocker, bn_adapt_section, is_called, tmp_path, schedule_params):
        test_desc = PSControllerTestDesc(model_creator=ThreeConvModel,
                                     algo_params={'width': {'min_width': 1, 'width_step': 1}},
                                     input_sizes=ThreeConvModel.INPUT_SIZE,
                                     )
        nncf_network, ctrl, nncf_config = prepare_test_model(test_desc)
        update_train_bn_adapt_section(nncf_config, is_called)
        bn_adapt_run_patch = mocker.patch(
            "nncf.common.initialization.batchnorm_adaptation.BatchnormAdaptationAlgorithm.run")
        mock_model = MockModel()
        mock_nncf_network = mocker.MagicMock(spec=NNCFNetwork)
        mock_width_handler = mocker.MagicMock(spec=ElasticWidthHandler)
        mock_depth_handler = mocker.MagicMock(spec=ElasticDepthHandler)
        mock_kernel_handler = mocker.MagicMock(spec=SingleElasticityHandler)
        handlers = OrderedDict({
            ElasticityDim.WIDTH: mock_width_handler,
            ElasticityDim.KERNEL: mock_kernel_handler,
            ElasticityDim.DEPTH: mock_depth_handler,
        })
        mock_handler = MultiElasticityHandler(handlers, mock_nncf_network)
        # pylint:disable=protected-access
        mock_elasticity_ctrl = mocker.stub()
        mock_elasticity_ctrl.multi_elasticity_handler = mock_handler
        lr_schedule_config = {}
        training_config = nncf_config.get('bootstrapNAS', {}).get('training', {})
        bn_adapt_params = training_config.get('batchnorm_adaptation', {})
        bn_adapt_algo_kwargs = get_bn_adapt_algo_kwargs(nncf_config, bn_adapt_params)
        bn_adaptation = BatchnormAdaptationAlgorithm(**bn_adapt_algo_kwargs) if bn_adapt_algo_kwargs else None
        training_algo = ProgressiveShrinkingController(mock_model, mock_elasticity_ctrl, bn_adaptation,
                                                       ProgressiveShrinkingBuilder.DEFAULT_PROGRESSIVITY,
                                                       schedule_params, lr_schedule_config)
        # Check prepare for validation
        training_algo.prepare_for_validation()
        # Check when setting stage
        desc = StageDescriptor([ElasticityDim.DEPTH, ElasticityDim.WIDTH], bn_adapt=is_called)
        training_algo.set_stage(desc)
        if is_called:
            bn_adapt_run_patch.assert_called()
        else:
            bn_adapt_run_patch.assert_not_called()