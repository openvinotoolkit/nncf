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
from collections import OrderedDict
from functools import partial
from typing import List

import pytest

from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_depth import ElasticDepthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.training.base_training import BNASTrainingAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler import GlobalLRScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler import StageLRScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.progressive_shrinking_builder import ProgressiveShrinkingBuilder
from nncf.experimental.torch.nas.bootstrapNAS.training.progressive_shrinking_controller import (
    ProgressiveShrinkingController,
)
from nncf.experimental.torch.nas.bootstrapNAS.training.scheduler import BootstrapNASScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.scheduler import NASSchedulerParams
from nncf.experimental.torch.nas.bootstrapNAS.training.stage_descriptor import DEFAULT_STAGE_LR_RATE
from nncf.experimental.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.torch.nncf_network import NNCFNetwork
from tests.torch.helpers import MockModel

LIST_STAGES__K_KW_KWD = [
    [ElasticityDim.KERNEL],
    [ElasticityDim.KERNEL, ElasticityDim.WIDTH],
    [ElasticityDim.KERNEL, ElasticityDim.WIDTH, ElasticityDim.DEPTH],
]

LIST_STAGES__K_KD_KDW = [
    [ElasticityDim.KERNEL],
    [ElasticityDim.KERNEL, ElasticityDim.DEPTH],
    [ElasticityDim.KERNEL, ElasticityDim.DEPTH, ElasticityDim.WIDTH],
]

SIMPLE_LIST_STAGE_DESCRIPTORS = [
    StageDescriptor(train_dims=[ElasticityDim.KERNEL], epochs=1),
    StageDescriptor(train_dims=[ElasticityDim.KERNEL, ElasticityDim.DEPTH], epochs=1, depth_indicator=1),
    StageDescriptor(train_dims=[ElasticityDim.KERNEL, ElasticityDim.DEPTH], epochs=1, depth_indicator=2),
    StageDescriptor(
        train_dims=[ElasticityDim.KERNEL, ElasticityDim.DEPTH, ElasticityDim.WIDTH],
        epochs=1,
        depth_indicator=2,
        reorg_weights=True,
        width_indicator=2,
    ),
    StageDescriptor(
        train_dims=[ElasticityDim.KERNEL, ElasticityDim.DEPTH, ElasticityDim.WIDTH],
        epochs=1,
        depth_indicator=2,
        reorg_weights=True,
        width_indicator=3,
    ),
]


@pytest.fixture(name="schedule_params", params=[SIMPLE_LIST_STAGE_DESCRIPTORS], ids=["simple_desc"])
def fixture_schedule_params(request):
    list_descriptors = request.param
    return NASSchedulerParams(list_descriptors)


LIST_DIMS__KDW = [ElasticityDim.KERNEL, ElasticityDim.DEPTH, ElasticityDim.WIDTH]


class TestScheduler:
    def test_get_stage(self, schedule_params: NASSchedulerParams, mocker):
        training_ctrl_mock = mocker.MagicMock(spec=BNASTrainingAlgorithm)
        training_ctrl_mock.lr_schedule_config = {}
        scheduler = BootstrapNASScheduler(training_ctrl_mock, schedule_params, LIST_DIMS__KDW, LIST_DIMS__KDW)
        optimizer_mock = mocker.stub()
        optimizer_mock.param_groups = [{"lr": 1}]
        scheduler.lr_scheduler = StageLRScheduler(optimizer_mock, 10)
        scheduler.epoch_step()
        ref_desc = StageDescriptor(
            train_dims=[ElasticityDim.KERNEL], epochs=1, init_lr=DEFAULT_STAGE_LR_RATE, epochs_lr=1
        )
        act_desc, act_idx = scheduler.get_current_stage_desc()
        assert ref_desc == act_desc
        assert act_idx == 0

        scheduler.epoch_step(next_epoch=2)
        ref_desc.train_dims.append(ElasticityDim.DEPTH)
        ref_desc.depth_indicator = 2
        act_desc, act_idx = scheduler.get_current_stage_desc()
        assert ref_desc == act_desc
        assert act_idx == 2

        scheduler.epoch_step()
        ref_desc.train_dims.append(ElasticityDim.WIDTH)
        ref_desc.reorg_weights = True
        ref_desc.width_indicator = 2
        act_desc, act_idx = scheduler.get_current_stage_desc()
        assert ref_desc == act_desc
        assert act_idx == 3

        scheduler.epoch_step()
        ref_desc.width_indicator = 3
        act_desc, act_idx = scheduler.get_current_stage_desc()
        assert ref_desc == act_desc
        assert act_idx == 4

    def test_epoch_step(self, schedule_params, mocker):
        mock_model = MockModel()
        mock_nncf_network = mocker.MagicMock(spec=NNCFNetwork)
        mock_width_handler = mocker.MagicMock(spec=ElasticWidthHandler)
        mock_depth_handler = mocker.MagicMock(spec=ElasticDepthHandler)
        mock_kernel_handler = mocker.MagicMock(spec=SingleElasticityHandler)
        handlers = OrderedDict(
            {
                ElasticityDim.WIDTH: mock_width_handler,
                ElasticityDim.KERNEL: mock_kernel_handler,
                ElasticityDim.DEPTH: mock_depth_handler,
            }
        )
        mock_handler = MultiElasticityHandler(handlers, mock_nncf_network)

        is_handler_enabled_map = mock_handler._is_handler_enabled_map
        mock_elasticity_ctrl = mocker.stub()
        mock_elasticity_ctrl.multi_elasticity_handler = mock_handler
        lr_schedule_config = {}
        training_algo = ProgressiveShrinkingController(
            mock_model,
            mock_elasticity_ctrl,
            mocker.stub(),
            ProgressiveShrinkingBuilder.DEFAULT_PROGRESSIVITY,
            schedule_params,
            lr_schedule_config,
            ZeroCompressionLoss(next(mock_model.parameters()).device),
        )
        scheduler = training_algo.scheduler
        lr_scheduler = GlobalLRScheduler(mocker.stub(), mocker.stub(), base_lr=None, num_epochs=None)
        scheduler.lr_scheduler = lr_scheduler
        scheduler.epoch_step()
        assert is_handler_enabled_map == {
            ElasticityDim.WIDTH: False,
            ElasticityDim.DEPTH: False,
            ElasticityDim.KERNEL: True,
        }

        scheduler.epoch_step()
        assert is_handler_enabled_map == {
            ElasticityDim.WIDTH: False,
            ElasticityDim.DEPTH: True,
            ElasticityDim.KERNEL: True,
        }
        assert mock_depth_handler.depth_indicator == 1

        scheduler.epoch_step()
        assert is_handler_enabled_map == {
            ElasticityDim.WIDTH: False,
            ElasticityDim.DEPTH: True,
            ElasticityDim.KERNEL: True,
        }
        assert mock_depth_handler.depth_indicator == 2

        scheduler.epoch_step()
        assert is_handler_enabled_map == {
            ElasticityDim.WIDTH: True,
            ElasticityDim.DEPTH: True,
            ElasticityDim.KERNEL: True,
        }
        mock_width_handler.reorganize_weights.assert_called()
        assert mock_width_handler.width_num_params_indicator == 2

        scheduler.epoch_step()
        assert is_handler_enabled_map == {
            ElasticityDim.WIDTH: True,
            ElasticityDim.DEPTH: True,
            ElasticityDim.KERNEL: True,
        }
        mock_width_handler.reorganize_weights.assert_called()
        assert mock_width_handler.width_num_params_indicator == 3

    def test_get_total_training_epochs(self, schedule_params, mocker):
        training_controller_mock = mocker.stub()
        training_controller_mock.lr_schedule_config = {}
        scheduler = BootstrapNASScheduler(
            training_controller_mock,
            schedule_params,
            available_elasticity_dims=LIST_DIMS__KDW,
            progressivity_of_elasticity=LIST_DIMS__KDW,
        )
        assert scheduler.get_total_training_epochs() == 5


class SchedulerTestDesc:
    def __init__(
        self,
        list_stage_dims: List[List[ElasticityDim]],
        progressivity_of_elasticity: List[ElasticityDim],
        available_elasticity_dims: List[ElasticityDim],
        name: str = "",
        error_in_scheduler: bool = False,
        error_in_builder: bool = False,
    ):
        self.list_stage_dims = list_stage_dims
        self.progressivity_of_elasticity = progressivity_of_elasticity
        self.available_elasticity_dims = available_elasticity_dims
        self.error_in_scheduler = error_in_scheduler
        self.error_in_builder = error_in_builder
        self.name = name

    def __str__(self):
        return self.name

    @property
    def scheduler_params(self) -> NASSchedulerParams:
        list_stage_descs = [
            {"train_dims": list(map(lambda x: x.value, stage_dims))} for stage_dims in self.list_stage_dims
        ]
        return NASSchedulerParams.from_config({"list_stage_descriptions": list_stage_descs})


LIST_SCHEDULER_DESCS = [
    SchedulerTestDesc(
        name="default",
        list_stage_dims=LIST_STAGES__K_KD_KDW,
        progressivity_of_elasticity=LIST_DIMS__KDW,
        available_elasticity_dims=LIST_DIMS__KDW,
    ),
    SchedulerTestDesc(
        name="wrong order in progressivity",
        list_stage_dims=LIST_STAGES__K_KW_KWD,
        progressivity_of_elasticity=LIST_DIMS__KDW,
        available_elasticity_dims=LIST_DIMS__KDW,
        error_in_scheduler=True,
    ),
    SchedulerTestDesc(
        name="limited progressivity",
        list_stage_dims=LIST_STAGES__K_KW_KWD,
        progressivity_of_elasticity=[ElasticityDim.KERNEL],
        available_elasticity_dims=LIST_DIMS__KDW,
        error_in_builder=True,
        error_in_scheduler=True,
    ),
    SchedulerTestDesc(
        name="limited enabled dims",
        list_stage_dims=LIST_STAGES__K_KW_KWD,
        progressivity_of_elasticity=LIST_DIMS__KDW,
        available_elasticity_dims=[ElasticityDim.KERNEL],
        error_in_scheduler=True,
    ),
    SchedulerTestDesc(
        name="limited progressivity and enabled dims",
        list_stage_dims=LIST_STAGES__K_KW_KWD,
        progressivity_of_elasticity=[ElasticityDim.KERNEL],
        available_elasticity_dims=[ElasticityDim.KERNEL],
        error_in_scheduler=True,
    ),
    SchedulerTestDesc(
        name="limited list stages",
        list_stage_dims=[[ElasticityDim.KERNEL]],
        progressivity_of_elasticity=LIST_DIMS__KDW,
        available_elasticity_dims=LIST_DIMS__KDW,
    ),
    SchedulerTestDesc(
        name="violated progressivity",
        list_stage_dims=LIST_STAGES__K_KW_KWD,
        progressivity_of_elasticity=[ElasticityDim.KERNEL, ElasticityDim.DEPTH, ElasticityDim.WIDTH],
        available_elasticity_dims=LIST_DIMS__KDW,
        error_in_scheduler=True,
    ),
    SchedulerTestDesc(
        name="order within stage doesn't matter",
        list_stage_dims=[
            [ElasticityDim.KERNEL],
            [ElasticityDim.DEPTH, ElasticityDim.KERNEL],
            [ElasticityDim.DEPTH, ElasticityDim.WIDTH, ElasticityDim.KERNEL],
        ],
        progressivity_of_elasticity=LIST_DIMS__KDW,
        available_elasticity_dims=LIST_DIMS__KDW,
    ),
    SchedulerTestDesc(
        name="new single dim on each stage",
        list_stage_dims=[
            [ElasticityDim.KERNEL],
            [ElasticityDim.DEPTH],
            [ElasticityDim.WIDTH],
        ],
        progressivity_of_elasticity=LIST_DIMS__KDW,
        available_elasticity_dims=LIST_DIMS__KDW,
        error_in_scheduler=True,
    ),
    SchedulerTestDesc(
        name="intermediate dim is not enabled",
        list_stage_dims=[
            [ElasticityDim.KERNEL],
            [ElasticityDim.DEPTH, ElasticityDim.KERNEL],
        ],
        progressivity_of_elasticity=[ElasticityDim.KERNEL, ElasticityDim.WIDTH, ElasticityDim.DEPTH],
        available_elasticity_dims=[ElasticityDim.KERNEL, ElasticityDim.DEPTH],
    ),
    SchedulerTestDesc(
        name="limited list stages started from intermediate",
        list_stage_dims=[[ElasticityDim.DEPTH], [ElasticityDim.DEPTH, ElasticityDim.WIDTH]],
        progressivity_of_elasticity=LIST_DIMS__KDW,
        available_elasticity_dims=LIST_DIMS__KDW,
    ),
]


@pytest.mark.parametrize("desc", LIST_SCHEDULER_DESCS, ids=map(str, LIST_SCHEDULER_DESCS))
class TestElasticityConsistency:
    def test_checks_on_scheduler_init(self, mocker, desc: SchedulerTestDesc):
        training_controller_mock = mocker.stub()
        training_controller_mock.lr_schedule_config = {}
        scheduler_fn = partial(
            BootstrapNASScheduler,
            training_controller_mock,
            desc.scheduler_params,
            progressivity_of_elasticity=desc.progressivity_of_elasticity,
            available_elasticity_dims=desc.available_elasticity_dims,
        )
        scheduler = scheduler_fn()
        if desc.error_in_scheduler:
            with pytest.raises(ValueError):
                _ = scheduler.list_stage_descriptors
        else:
            _ = scheduler.list_stage_descriptors

    def test_progressivity_vs_enabled_dims(self, desc: SchedulerTestDesc):
        builder_fn = partial(
            ProgressiveShrinkingBuilder.check_elasticity_dims_consistency,
            desc.available_elasticity_dims,
            desc.progressivity_of_elasticity,
        )
        if desc.error_in_builder:
            with pytest.raises(ValueError):
                builder_fn()
        else:
            builder_fn()
