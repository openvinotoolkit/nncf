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
from enum import IntEnum
from typing import Any, Dict, Optional

import torch

from nncf.common.logging import nncf_logger
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.schedulers import PolynomialDecaySchedule
from nncf.config.schemata.experimental_schema import MOVEMENT_ENABLE_STRUCTURED_MASKING
from nncf.config.schemata.experimental_schema import MOVEMENT_FINAL_IMPORTANCE_THRESHOLD
from nncf.config.schemata.experimental_schema import MOVEMENT_POWER


class MovementSchedulerStage(IntEnum):
    """
    Describes the current stage of training with movement sparsity.
    """

    PRE_WARMUP = 0
    IN_WARMUP = 1
    POST_WARMUP = 2


class MovementSchedulerParams:
    """
    Stores the params to initialize the scheduler of movement sparsity.
    """

    def __init__(
        self,
        warmup_start_epoch: int,
        warmup_end_epoch: int,
        importance_regularization_factor: float,
        enable_structured_masking: bool = MOVEMENT_ENABLE_STRUCTURED_MASKING,
        init_importance_threshold: Optional[float] = None,
        final_importance_threshold: float = MOVEMENT_FINAL_IMPORTANCE_THRESHOLD,
        power: float = MOVEMENT_POWER,
        steps_per_epoch: Optional[int] = None,
    ):
        """
        Initializes and validates the params for scheduler.

        :param warmup_start_epoch: Index of the starting epoch (inclusive) for warmup stage.
        :param warmup_end_epoch: Index of the end epoch (exclusive) for warmup stage.
        :param importance_regularization_factor: The regularization factor on weight importance scores.
        :param enable_structured_masking: Whether to do structured mask resolution after warmup stage.
        :param init_importance_threshold: The initial value of importance threshold during warmup stage.
        :param final_importance_threshold: The final value of importance threshold during warmup stage.
        :param power: The power value of polynomial decay for threshold update during warmup stage.
        :param steps_per_epoch: Number of training steps in one epoch.
        """

        if steps_per_epoch is None and warmup_start_epoch < 1:
            raise ValueError(
                "`warmup_start_epoch` must be >= 1 to enable the auto calculation of "
                "`steps_per_epoch`. Please either change `warmup_start_epoch` to a larger "
                "number or specify `steps_per_epoch` in the config."
            )

        if warmup_start_epoch < 0 or warmup_end_epoch <= warmup_start_epoch:
            raise ValueError("Movement sparsity requires 0 <= warmup_start_epoch < warmup_end_epoch.")

        if importance_regularization_factor < 0:
            raise ValueError("`importance_regularization_factor` should not be a negative number.")

        if init_importance_threshold is not None and init_importance_threshold >= final_importance_threshold:
            nncf_logger.warning(
                "`init_importance_threshold` is equal to or greater than "
                "`final_importance_threshold`. Movement sparsity may not work as expected."
            )

        self.warmup_start_epoch = warmup_start_epoch
        self.warmup_end_epoch = warmup_end_epoch
        self.importance_regularization_factor = importance_regularization_factor
        self.enable_structured_masking = enable_structured_masking
        self.init_importance_threshold = init_importance_threshold
        self.final_importance_threshold = final_importance_threshold
        self.power = power
        self.steps_per_epoch = steps_per_epoch

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "MovementSchedulerParams":
        """
        Initialize `MovementSchedulerParams` object from the config in dict format.

        :param params: A dict that specifies the parameters of movement sparsity scheduler.
        :return: A `MovementSchedulerParams` object that stores the parameters from `params`.
        """
        warmup_start_epoch: int = params.get("warmup_start_epoch")
        warmup_end_epoch: int = params.get("warmup_end_epoch")
        importance_regularization_factor: float = params.get("importance_regularization_factor")
        enable_structured_masking: bool = params.get("enable_structured_masking", MOVEMENT_ENABLE_STRUCTURED_MASKING)
        init_importance_threshold: Optional[float] = params.get("init_importance_threshold")
        final_importance_threshold: float = params.get(
            "final_importance_threshold", MOVEMENT_FINAL_IMPORTANCE_THRESHOLD
        )
        power: float = params.get("power", MOVEMENT_POWER)
        steps_per_epoch: Optional[int] = params.get("steps_per_epoch")

        if None in [warmup_start_epoch, warmup_end_epoch, importance_regularization_factor]:
            raise ValueError(
                "`warmup_start_epoch`, `warmup_start_epoch` and `importance_regularization_factor` "
                "are required in config for Movement Sparsity."
            )

        return cls(
            warmup_start_epoch=warmup_start_epoch,
            warmup_end_epoch=warmup_end_epoch,
            importance_regularization_factor=importance_regularization_factor,
            enable_structured_masking=enable_structured_masking,
            init_importance_threshold=init_importance_threshold,
            final_importance_threshold=final_importance_threshold,
            power=power,
            steps_per_epoch=steps_per_epoch,
        )


class MovementPolynomialThresholdScheduler(BaseCompressionScheduler):
    """
    Movement Sparsity scheduler with a polynomial decay schedule for importance
    threshold and importance regularization factor.

    The scheduler will update importance threshold and importance regularization
    factor per optimizer step. Parameter `steps_per_epoch` should be provided
    in config for the per step calculation. If not provided, the scheduler will
    use the first epoch to calculate `steps_per_epoch` parameter. In this case,
    parameter `warmup_start_epoch` must be equal to or larger than 1, and the
    scheduler will start calculation only after `steps_per_epoch` is calculated.
    """

    def __init__(self, controller: "MovementSparsityController", params: MovementSchedulerParams):  # noqa: F821
        """
        Initializes a movement sparsity scheduler with a polynomial decay schedule.

        :param controller: Movement sparsity algorithm controller.
        :param params: Parameters for the scheduler.
        """
        super().__init__()
        self._controller = controller
        self._params = params
        self._schedule = PolynomialDecaySchedule(
            initial_value=0.0,
            target_value=1.0,
            target_epoch=(self._params.warmup_end_epoch - self._params.warmup_start_epoch),
            power=self._params.power,
            concave=True,
        )
        self._steps_per_epoch = self._params.steps_per_epoch
        self._init_importance_threshold = self._params.init_importance_threshold
        self._cached_importance_threshold_lambda = None
        self._is_controller_frozen = False
        self._steps_in_current_epoch = 0
        self._should_skip = False

    @property
    def current_stage(self) -> MovementSchedulerStage:
        if self._steps_per_epoch is None or self.current_step < int(
            self._params.warmup_start_epoch * self._steps_per_epoch
        ):
            return MovementSchedulerStage.PRE_WARMUP
        if self.current_step < int(self._params.warmup_end_epoch * self._steps_per_epoch):
            return MovementSchedulerStage.IN_WARMUP
        return MovementSchedulerStage.POST_WARMUP

    @property
    def current_importance_regularization_factor(self) -> float:
        """
        Calculates the value of importance regularization factor per the current state. The factor
        stays zero before warmup stage, and gradually increases during warmup, and stays at
        the fixed value after warmup.

        :return: The value of importance regularization factor at the current step.
        """
        current_stage = self.current_stage
        if current_stage == MovementSchedulerStage.PRE_WARMUP:
            return 0.0
        if current_stage == MovementSchedulerStage.IN_WARMUP:
            return self._calc_current_scheduled_value(0.0, self._params.importance_regularization_factor)
        return self._params.importance_regularization_factor

    @property
    def current_importance_threshold(self) -> float:
        """
        Calculates the value of importance threshold per the current state. The threshold
        stays `-math.inf` before warmup stage, and gradually increases from `self._init_importance_threshold`
        during warmup, and finally stays fixed at the specified final value.

        :return: The value of importance threshold at the current step.
        """
        current_stage = self.current_stage
        if current_stage == MovementSchedulerStage.PRE_WARMUP:
            return -math.inf
        if current_stage == MovementSchedulerStage.IN_WARMUP:
            assert self._init_importance_threshold is not None
            return self._calc_current_scheduled_value(
                self._init_importance_threshold, self._params.final_importance_threshold
            )
        return self._params.final_importance_threshold

    @property
    def enable_structured_masking(self) -> bool:
        return self._params.enable_structured_masking

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        super().epoch_step(next_epoch)
        self._maybe_should_skip()
        self._steps_in_current_epoch = 0

    def step(self, next_step: Optional[int] = None) -> None:
        super().step(next_step)
        self._steps_in_current_epoch += 1
        if self._should_skip:
            return
        self._schedule_controller()

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["_init_importance_threshold"] = self._init_importance_threshold
        state["_steps_per_epoch"] = self._steps_per_epoch
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        super().load_state(state)
        self._init_importance_threshold = state["_init_importance_threshold"]
        self._steps_per_epoch = state["_steps_per_epoch"]
        if self._steps_per_epoch is None:  # It is the first epoch and `steps_per_epoch` not specified
            self._steps_in_current_epoch = self._current_step + 1
            self._should_skip = True
        else:
            self._steps_in_current_epoch = self._current_step % self._steps_per_epoch + 1

    def _schedule_controller(self):
        """
        Asks and updates the controller during training steps. It (1) updates the importance threshold
        and importance regularization factor in the operand at each step; (2) freezes the controller
        after warmup; (3) calculates the initial importance threshold if unspecified; (4) conducts
        structured masking if supported.
        """
        if self._init_importance_threshold is None and self.current_stage == MovementSchedulerStage.IN_WARMUP:
            adaptive_init_threshold = self._calc_init_threshold_from_controller(target_sparsity=0.001)
            nncf_logger.info(
                "Movement sparsity automatically calculates `init_importance_threshold` as "
                f"{adaptive_init_threshold} so that warmup starts from ~0.1% linear layer sparsity."
            )
            if adaptive_init_threshold >= self._params.final_importance_threshold:
                nncf_logger.warning(
                    "The auto-calculated `init_importance_threshold` is equal to or greater than "
                    "`final_importance_threshold`. Movement sparsity may not work as expected."
                )
            self._init_importance_threshold = adaptive_init_threshold
        if self.current_stage == MovementSchedulerStage.POST_WARMUP and (not self._is_controller_frozen):
            if self._params.enable_structured_masking:
                self._controller.reset_independent_structured_mask()
                self._controller.resolve_structured_mask()
                self._controller.populate_structured_mask()
            self._controller.freeze()
            self._is_controller_frozen = True
        self._update_operand_importance_threshold_and_factor()

    def _calc_current_scheduled_value(self, start_value: float, end_value: float) -> float:
        assert self.current_stage == MovementSchedulerStage.IN_WARMUP
        assert self._steps_per_epoch is not None
        schedule_current_step = self.current_step - int(self._params.warmup_start_epoch * self._steps_per_epoch)
        schedule_epoch = schedule_current_step // self._steps_per_epoch
        schedule_step = schedule_current_step % self._steps_per_epoch
        scale = self._schedule(schedule_epoch, schedule_step, self._steps_per_epoch)
        return start_value + scale * (end_value - start_value)

    @torch.no_grad()
    def _calc_init_threshold_from_controller(self, target_sparsity: float = 0.001) -> float:
        # Calculate the k-th smallest value over all importance scores as the initial importance threshold
        # so that roughly k weight elements are masked and thus target sparsity is achieved. We conduct this on
        # CPU to (1) limit GPU memory usage; (2) avoid the non-deterministic behavior of `torch.Tensor.kthvalue`
        # on CUDA.
        assert 0.0 <= target_sparsity < 1.0
        importance_tensors = []
        for minfo in self._controller.sparsified_module_info:
            operand = minfo.operand
            weight = operand.get_importance(is_bias=False, expanded=True)
            importance_tensors.append(weight.detach().cpu().view(-1))
            if operand.prune_bias:
                bias = operand.get_importance(is_bias=True, expanded=True)
                importance_tensors.append(bias.detach().cpu().view(-1))
        all_importances = torch.cat(importance_tensors)
        k = min(all_importances.numel(), int(all_importances.numel() * target_sparsity) + 1)  # k starts from 1
        return all_importances.kthvalue(k).values.item()

    def _update_operand_importance_threshold_and_factor(self):
        current = (self.current_importance_threshold, self.current_importance_regularization_factor)
        if current != self._cached_importance_threshold_lambda:
            for minfo in self._controller.sparsified_module_info:
                minfo.operand.importance_threshold = self.current_importance_threshold
                minfo.operand.importance_regularization_factor = self.current_importance_regularization_factor
        self._cached_importance_threshold_lambda = current

    def _maybe_should_skip(self) -> None:
        """
        Checks if the first epoch (with index 0) should be skipped to calculate
        the steps per epoch. If the skip is needed, then the internal state
        of the scheduler object will not be changed.
        """
        self._should_skip = False

        if self._steps_per_epoch is None and self._steps_in_current_epoch > 0:
            self._steps_per_epoch = self._steps_in_current_epoch

        if (
            self._steps_per_epoch is not None
            and self._steps_in_current_epoch > 0
            and self._steps_per_epoch != self._steps_in_current_epoch
        ):
            raise Exception(
                "Actual steps per epoch and steps per epoch from the scheduler "
                "parameters are different. Scheduling may be incorrect."
            )

        if self._steps_per_epoch is None:
            self._should_skip = True
            nncf_logger.info(
                "Movement sparsity scheduler updates importance threshold and regularization"
                "factor per optimizer step, but steps_per_epoch was not set in config. Will "
                "measure the actual steps per epoch as signaled by a .epoch_step() call."
            )
