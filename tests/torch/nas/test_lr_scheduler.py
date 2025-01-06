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

from copy import copy

import pytest

from nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler import GlobalLRScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler import LRSchedulerParams
from nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler import StageLRScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler import calc_learning_rate

LR_SCHEDULER_PARAMS = LRSchedulerParams.from_dict(
    {
        "num_steps_in_epoch": 10,
        "base_lr": 2.5e-6,
        "num_epochs": 5,
        "warmup_epochs": 2,
    }
)


class TestLRScheduler:
    def test_warmup_lr(self, mocker):
        warmup_lr = mocker.patch(
            "nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler.warmup_adjust_learning_rate"
        )
        optimizer = mocker.stub()
        optimizer.param_groups = [{"lr": 0}]
        lr_scheduler = GlobalLRScheduler.from_config(optimizer, LR_SCHEDULER_PARAMS)
        lr_scheduler.epoch_step(0)
        lr_scheduler.step()
        warmup_lr.assert_called()

    def test_adjust_lr(self, mocker):
        adjust_lr = mocker.patch("nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler.adjust_learning_rate")
        optimizer = mocker.stub()
        optimizer.param_groups = [{"lr": 0}]
        params = copy(LR_SCHEDULER_PARAMS)
        params.warmup_epochs = 0
        lr_scheduler = GlobalLRScheduler.from_config(optimizer, params)
        lr_scheduler.epoch_step(0)
        lr_scheduler.step()
        adjust_lr.assert_called()

    def test_warmup_to_regular(self, mocker):
        warmup_lr = mocker.patch(
            "nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler.warmup_adjust_learning_rate"
        )
        adjust_lr = mocker.patch("nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler.adjust_learning_rate")
        optimizer = mocker.stub()
        optimizer.param_groups = [{"lr": 0}]
        params = copy(LR_SCHEDULER_PARAMS)
        params.warmup_epochs = 1
        lr_scheduler = GlobalLRScheduler.from_config(optimizer, params)
        lr_scheduler.epoch_step(0)
        lr_scheduler.step()
        warmup_lr.assert_called()
        lr_scheduler.epoch_step()
        lr_scheduler.step()
        adjust_lr.assert_called()

    def test_reset(self, mocker):
        optimizer = mocker.stub()
        optimizer.param_groups = [{"lr": 1}]
        lr_scheduler = StageLRScheduler.from_config(optimizer, LR_SCHEDULER_PARAMS)
        lr_scheduler._init_lr = 2.5e-4
        lr_scheduler._num_epochs = 10
        assert lr_scheduler.current_epoch == -1
        assert lr_scheduler.current_step == -1
        lr_scheduler.epoch_step()
        lr_scheduler.step()
        assert lr_scheduler.current_epoch == 0
        assert lr_scheduler.current_step == 0
        lr_scheduler.epoch_step()
        lr_scheduler.step()
        assert lr_scheduler.current_epoch == 1
        assert lr_scheduler.current_step == 1
        lr_scheduler.reset(LR_SCHEDULER_PARAMS.base_lr, LR_SCHEDULER_PARAMS.num_epochs)
        assert lr_scheduler.current_epoch == 0
        assert lr_scheduler.current_step == 0

    def get_global_lr(self, lr_scheduler):
        step_from_epoch_start = lr_scheduler.current_step - (
            lr_scheduler.current_epoch * (lr_scheduler._num_steps_in_epoch + 1)
        )
        print(
            lr_scheduler.current_epoch < lr_scheduler._warmup_epochs and lr_scheduler.current_epoch != -1,
            lr_scheduler.current_epoch,
            lr_scheduler._warmup_epochs,
        )
        if lr_scheduler.current_epoch < lr_scheduler._warmup_epochs and lr_scheduler.current_epoch != -1:
            print("Warmup")
            t_cur = lr_scheduler.current_epoch * lr_scheduler._num_steps_in_epoch + step_from_epoch_start + 1
            new_lr = (
                t_cur
                / (lr_scheduler._warmup_epochs * lr_scheduler._num_steps_in_epoch)
                * (lr_scheduler._base_lr - lr_scheduler._warmup_lr)
                + lr_scheduler._warmup_lr
            )
        else:
            new_lr = calc_learning_rate(
                lr_scheduler.current_epoch - lr_scheduler._warmup_epochs,
                lr_scheduler._base_lr,
                lr_scheduler._num_epochs,
                step_from_epoch_start,
                lr_scheduler._num_steps_in_epoch,
                "cosine",
            )
        return new_lr

    def get_stage_lr(self, lr_scheduler):
        step_from_epoch_start = lr_scheduler.current_step - (
            lr_scheduler.current_epoch * (lr_scheduler._num_steps_in_epoch + 1)
        )
        return calc_learning_rate(
            lr_scheduler.current_epoch,
            lr_scheduler._init_lr,
            lr_scheduler._num_epochs,
            step_from_epoch_start,
            lr_scheduler._num_steps_in_epoch,
            "cosine",
        )

    def test_lr_value_update(self, mocker):
        optimizer = mocker.stub()
        optimizer.param_groups = [{"lr": 3.4e-4}]
        lr_scheduler = StageLRScheduler.from_config(optimizer, LR_SCHEDULER_PARAMS)
        lr_scheduler._init_lr = 2.5e-4
        lr_scheduler._num_epochs = 10
        for _ in range(lr_scheduler._num_epochs):
            lr_scheduler.epoch_step()
            for _ in range(lr_scheduler._num_steps_in_epoch):
                lr_scheduler.step()
                assert optimizer.param_groups[0]["lr"] == pytest.approx(self.get_stage_lr(lr_scheduler))

        optimizer.param_groups = [{"lr": 3.4e-4}]
        lr_scheduler = GlobalLRScheduler.from_config(optimizer, LR_SCHEDULER_PARAMS)
        for _ in range(lr_scheduler._num_epochs):
            lr_scheduler.epoch_step()
            for _ in range(lr_scheduler._num_steps_in_epoch):
                lr_scheduler.step()
                assert optimizer.param_groups[0]["lr"] == pytest.approx(self.get_global_lr(lr_scheduler))
