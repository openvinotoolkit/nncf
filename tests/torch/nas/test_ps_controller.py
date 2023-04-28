# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, NamedTuple

import pytest

from nncf import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.experimental.torch.nas.bootstrapNAS import EpochBasedTrainingAlgorithm
from nncf.torch.model_creation import create_nncf_network
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.nas.helpers import move_model_to_cuda_if_available
from tests.torch.nas.models.synthetic import ThreeConvModel
from tests.torch.nas.test_scheduler import fixture_schedule_params  # pylint: disable=unused-import


class PSControllerTestDesc(NamedTuple):
    model_creator: Any
    blocks_to_skip: List[List[str]] = None
    input_sizes: List[int] = [1, 3, 32, 32]
    algo_params: Dict = {}
    name: str = None
    mode: str = "auto"

    def __str__(self):
        if hasattr(self.model_creator, "__name__"):
            name = self.model_creator.__name__
        elif self.name is not None:
            name = self.name
        else:
            name = "NOT_DEFINED"
        return name


def prepare_test_model(ps_ctrl_desc, bn_adapt_section_is_called):
    config = {
        "input_info": {"sample_size": ps_ctrl_desc.input_sizes},
        "bootstrapNAS": {
            "training": {
                "batchnorm_adaptation": {"num_bn_adaptation_samples": 2},
            },
        },
    }
    nncf_config = NNCFConfig.from_dict(config)
    update_train_bn_adapt_section(nncf_config, bn_adapt_section_is_called)
    bn_adapt_args = BNAdaptationInitArgs(data_loader=create_ones_mock_dataloader(nncf_config))
    nncf_config.register_extra_structs([bn_adapt_args])
    model = ps_ctrl_desc.model_creator()
    move_model_to_cuda_if_available(model)
    return model, bn_adapt_args, nncf_config


def update_train_bn_adapt_section(nncf_config, bn_adapt_section_is_called):
    if not bn_adapt_section_is_called:
        nncf_config["bootstrapNAS"]["training"]["batchnorm_adaptation"]["num_bn_adaptation_samples"] = 0


# pylint: disable=protected-access
class TestProgressiveTrainingController:
    @pytest.mark.parametrize(
        "bn_adapt_section_is_called",
        [False, True],
        ids=["section_with_zero_num_samples", "section_with_non_zero_num_samples"],
    )
    def test_bn_adapt(self, mocker, bn_adapt_section_is_called, schedule_params):
        test_desc = PSControllerTestDesc(
            model_creator=ThreeConvModel,
            algo_params={"width": {"min_width": 1, "width_step": 1}},
            input_sizes=ThreeConvModel.INPUT_SIZE,
        )
        bn_adapt_run_patch = mocker.patch(
            "nncf.common.initialization.batchnorm_adaptation.BatchnormAdaptationAlgorithm.run"
        )
        model, _, nncf_config = prepare_test_model(test_desc, bn_adapt_section_is_called)
        model = create_nncf_network(model, nncf_config)

        training_algorithm = EpochBasedTrainingAlgorithm.from_config(model, nncf_config)
        training_algorithm._training_ctrl.prepare_for_validation()
        if bn_adapt_section_is_called:
            bn_adapt_run_patch.assert_called()
        else:
            bn_adapt_run_patch.assert_not_called()
