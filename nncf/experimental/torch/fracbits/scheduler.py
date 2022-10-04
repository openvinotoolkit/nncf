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

from typing import Callable

from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.torch.fracbits.params import FracBitsSchedulerParams


class FracBitsQuantizationScheduler(BaseCompressionScheduler):
    def __init__(self, freeze_callback: Callable, params: FracBitsSchedulerParams):
        super().__init__()
        self._freeze_epoch = params.freeze_epoch
        self._freeze_callback = freeze_callback

        if self._freeze_epoch < 0:
            nncf_logger.warning(f"freeze_epoch={self._freeze_epoch} is less than 0. Don't freeze fractional bit widths")

    def epoch_step(self, next_epoch=None):
        super().epoch_step(next_epoch)
        if self._current_epoch == self._freeze_epoch:
            nncf_logger.info(f"Current epoch is {self._current_epoch}. Freeze fractional bit widths.")
            self._freeze_callback()
