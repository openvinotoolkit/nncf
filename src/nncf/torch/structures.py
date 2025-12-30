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
"""
PyTorch-specific structure definitions for passing arguments into certain NNCF calls.
"""

from typing import Optional

from nncf.common.utils.api_marker import api


@api()
class ExecutionParameters:
    """
    Parameters that are necessary for distributed training of the model.

    :param cpu_only: whether cpu-only mode is using for training
    :param current_gpu: id of GPU that should be used for training (if only one of all is used)
    """

    def __init__(self, cpu_only: bool, current_gpu: Optional[int]):
        self.cpu_only = cpu_only
        self.current_gpu = current_gpu
