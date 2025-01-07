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
from abc import abstractmethod
from typing import Any, Dict, List, Optional, TypeVar

from nncf.common.schedulers import BaseCompressionScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor

OptimizerType = TypeVar("OptimizerType")


def adjust_learning_rate(
    optimizer: OptimizerType,
    epoch: float,
    init_lr: float,
    epochs: float,
    batch: float = 0,
    n_batch: float = 0,
    lr_schedule_type: str = "cosine",
):
    new_lr = calc_learning_rate(epoch, init_lr, epochs, batch, n_batch, lr_schedule_type)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


def warmup_adjust_learning_rate(
    optimizer: OptimizerType,
    init_lr: float,
    t_total: float,
    n_batch: float,
    epoch: float,
    batch: float = 0,
    warmup_lr: float = 0,
):
    t_cur = epoch * n_batch + batch + 1
    new_lr = t_cur / t_total * (init_lr - warmup_lr) + warmup_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


def calc_learning_rate(
    epoch: float,
    init_lr: float,
    n_epochs: float,
    batch: float = 0,
    n_batch: float = 0,
    lr_schedule_type: str = "cosine",
):
    if lr_schedule_type == "cosine":
        t_total = n_epochs * n_batch
        t_cur = epoch * n_batch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError("do not support: %s" % lr_schedule_type)
    return lr


class LRSchedulerParams:
    """
    Storage class for LR Scheduler parameters
    """

    def __init__(
        self,
        num_steps_in_epoch: float,
        base_lr: float = 3.4e-4,
        num_epochs: float = 0,
        warmup_epochs: float = 0,
        warmup_lr: float = 0,
    ):
        """
        Initializes storage class for learning rate scheduler parameters

        :param num_steps_in_epoch:
        :param base_lr:
        :param num_epochs:
        :param warmup_epochs:
        :param warmup_lr:
        """
        self.num_steps_in_epoch = num_steps_in_epoch
        self.base_lr = base_lr
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr

    @classmethod
    def from_dict(cls, lr_scheduler_config: Dict[str, Any]) -> "LRSchedulerParams":
        """
        Initialize learning rate scheduler parameters storage clas from Dict.
        :param lr_scheduler_config: Dict with parameters of learning rate scheduler.
        :return:
        """
        num_steps_in_epoch = lr_scheduler_config.get("num_steps_in_epoch", 0)
        base_lr = lr_scheduler_config.get("base_lr", 3.4e-4)
        num_epochs = lr_scheduler_config.get("num_epochs", 0)
        warmup_epochs = lr_scheduler_config.get("warmup_epochs", 0)
        warmup_lr = lr_scheduler_config.get("warmup_lr", 3.4e-4)
        return cls(num_steps_in_epoch, base_lr, num_epochs, warmup_epochs, warmup_lr)


class BaseLRScheduler(BaseCompressionScheduler):
    """
    Base class for the learning rate scheduler
    """

    def __init__(self, optimizer: OptimizerType, num_steps_in_epoch: float):
        super().__init__()
        self._optimizer = optimizer
        self._num_steps_in_epoch = num_steps_in_epoch

    @abstractmethod
    def stage_step(self, stage_desc: StageDescriptor):
        pass

    @classmethod
    def from_state(cls, state: Dict[str, Any], optimizer: OptimizerType):
        return cls(optimizer, **state)

    def get_last_lr(self) -> List[Any]:
        return [group["lr"] for group in self._optimizer.param_groups]


class GlobalLRScheduler(BaseLRScheduler):
    """
    Global LR scheduler prevents LR adjustments per stage.
    """

    def __init__(
        self,
        optimizer: OptimizerType,
        num_steps_in_epoch: float,
        *,
        base_lr: float,
        num_epochs: float,
        warmup_epochs: float = 0,
        warmup_lr: float = 3.4e-4,
    ):
        super().__init__(optimizer, num_steps_in_epoch)
        self._base_lr = base_lr
        self._num_epochs = num_epochs
        self._warmup_epochs = warmup_epochs
        self._warmup_lr = warmup_lr

    @classmethod
    def from_config(cls, optimizer: OptimizerType, params: "LRSchedulerParams"):
        return cls(optimizer, **params.__dict__)

    def stage_step(self, stage_desc: StageDescriptor):
        # do nothing
        pass

    def step(self, next_step: Optional[int] = None) -> None:
        super().step(next_step)
        step_from_epoch_start = self.current_step - (self.current_epoch * (self._num_steps_in_epoch + 1))
        if self.current_epoch < self._warmup_epochs and self.current_epoch != -1:
            warmup_adjust_learning_rate(
                optimizer=self._optimizer,
                init_lr=self._base_lr,
                t_total=self._warmup_epochs * self._num_steps_in_epoch,
                n_batch=self._num_steps_in_epoch,
                epoch=self.current_epoch,
                batch=step_from_epoch_start,
                warmup_lr=self._warmup_lr,
            )
        else:
            adjust_learning_rate(
                optimizer=self._optimizer,
                epoch=self.current_epoch - self._warmup_epochs,
                init_lr=self._base_lr,
                epochs=self._num_epochs,
                batch=step_from_epoch_start,
                n_batch=self._num_steps_in_epoch,
                lr_schedule_type="cosine",
            )

    def get_state(self) -> Dict[str, Any]:
        state_dict = {
            "num_steps_in_epoch": self._num_steps_in_epoch,
            "base_lr": self._base_lr,
            "num_epochs": self._num_epochs,
            "warmup_epochs": self._warmup_epochs,
            "warmup_lr": self._warmup_lr,
        }
        return state_dict


class StageLRScheduler(BaseLRScheduler):
    """
    Stage learning rate scheduler. Allows adjustment of the learning rate at a stage transition.
    """

    def __init__(self, optimizer: OptimizerType, num_steps_in_epoch: float):
        super().__init__(optimizer, num_steps_in_epoch)
        self._init_lr = None
        self._num_epochs = None

    @classmethod
    def from_config(cls, optimizer: OptimizerType, params: LRSchedulerParams):
        return cls(optimizer, params.num_steps_in_epoch)

    def stage_step(self, stage_desc: StageDescriptor):
        self.reset(stage_desc.init_lr, stage_desc.epochs_lr)

    def step(self, next_step: Optional[int] = None) -> None:
        super().step(next_step)
        step_from_epoch_start = self.current_step - (self.current_epoch * (self._num_steps_in_epoch + 1))
        adjust_learning_rate(
            optimizer=self._optimizer,
            epoch=self.current_epoch,
            init_lr=self._init_lr,
            epochs=self._num_epochs,
            batch=step_from_epoch_start,
            n_batch=self._num_steps_in_epoch,
            lr_schedule_type="cosine",
        )

    def get_state(self) -> Dict[str, Any]:
        state_dict = {
            "num_steps_in_epoch": self._num_steps_in_epoch,
            "init_lr": self._init_lr,
            "num_epochs": self._num_epochs,
        }
        return state_dict

    def reset(self, init_lr: float, num_epochs: float):
        """
        Resets the learning rate for the current stage

        :param init_lr:
        :param num_epochs:
        :return:
        """
        self._num_epochs = num_epochs
        self._init_lr = init_lr
        self.epoch_step(0)
        self.step(0)
