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

import nncf


def multi_step_ratio_scheduler(epoch: int, *, steps: dict[int, int]) -> float:
    """
    Multi-step ratio scheduler.

    This function returns a ratio based on the current epoch and a dictionary of milestone steps.

    :param epoch: The current training epoch (must be >= 0).
    :param steps: A dictionary where keys are epoch thresholds (must include 0) and
        values are the corresponding schedule values. For example: {0: 1.0, 10: 0.5, 20: 0.1}
    :return: The scheduled value corresponding to the given epoch.
    """
    if epoch < 0:
        msg = "Epoch must be a non-negative integer."
        raise nncf.InternalError(msg)

    if 0 not in steps:
        msg = "Steps dictionary must include an entry for epoch 0."
        raise nncf.InternalError(msg)

    if any(ratio < 0 or ratio >= 1.0 for ratio in steps.values()):
        msg = "All ratio values in steps must be in the range [0.0, 1.0)."
        raise nncf.InternalError(msg)

    for threshold in sorted(steps.keys(), reverse=True):
        if epoch >= threshold:
            return steps[threshold]

    msg = "Unexpected state: no matching step found."
    raise nncf.InternalError(msg)


def exponential_ratio_scheduler(epoch: int, *, initial_ratio: float, target_ratio: float, target_epoch: int) -> float:
    """
    Exponential ratio scheduler.

    This scheduler smoothly transitions a ratio from an initial value to a target value
    over a specified number of epochs using an exponential progression.

    The ratio increases exponentially according to the following formula:
        ratio(epoch) = 1 - (1 - initial_ratio) * ((1 - target_ratio) / (1 - initial_ratio)) ** (epoch / target_epoch)

    :param epoch: Current epoch.
    :param initial_ratio: Starting ratio at epoch 0.
    :param target_ratio: Final ratio at target_epoch.
    :param target_epoch: Epoch at which the ratio should reach the target_ratio..
    :return: The scheduled value corresponding to the given epoch.
    """
    if epoch < 0:
        msg = "Epoch must be a non-negative integer."
        raise nncf.InternalError(msg)

    if initial_ratio < 0 or initial_ratio >= 1:
        msg = "Initial ratio should be in range [0, 1)."
        raise nncf.InternalError(msg)

    if target_ratio < 0 or target_ratio >= 1:
        msg = "Target ratio should be in range [0, 1)."
        raise nncf.InternalError(msg)

    if initial_ratio >= target_ratio:
        msg = "Initial sparsity should be less than target sparsity."
        raise nncf.InternalError(msg)

    if target_epoch <= 0:
        msg = "Total epochs should be a non-negative integer."
        raise nncf.InternalError(msg)

    if epoch == 0:
        return initial_ratio
    if epoch >= target_epoch:
        return target_ratio

    d_init = 1 - initial_ratio
    d_target = 1 - target_ratio
    d = d_init * float((d_target / d_init) ** (epoch / target_epoch))
    ratio = 1 - d
    return min(max(initial_ratio, ratio), target_epoch)
