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
from copy import deepcopy
from typing import Any, Dict, List, Optional

from nncf.experimental.torch.sparsity.movement.scheduler import MovementSchedulerParams


def convert_scheduler_params_to_dict(params: MovementSchedulerParams) -> Dict[str, Any]:
    result = {}
    for key, value in params.__dict__.items():
        if value is not None:
            result[key] = value
    return result


class MovementAlgoConfig:
    default_scheduler_params = MovementSchedulerParams(
        warmup_start_epoch=1,
        warmup_end_epoch=3,
        init_importance_threshold=-1.0,
        importance_regularization_factor=0.1,
        steps_per_epoch=4,
    )

    def __init__(
        self,
        scheduler_params: Optional[MovementSchedulerParams] = None,
        sparse_structure_by_scopes: Optional[List[Dict]] = None,
        ignored_scopes: Optional[List[str]] = None,
        compression_lr_multiplier: Optional[float] = None,
    ):
        self.scheduler_params = scheduler_params or deepcopy(MovementAlgoConfig.default_scheduler_params)
        self.sparse_structure_by_scopes = sparse_structure_by_scopes or []
        self.ignored_scopes = ignored_scopes or []
        self.compression_lr_multiplier = compression_lr_multiplier

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "algorithm": "movement_sparsity",
            "params": convert_scheduler_params_to_dict(self.scheduler_params),
            "sparse_structure_by_scopes": self.sparse_structure_by_scopes,
            "ignored_scopes": self.ignored_scopes,
        }
        if self.compression_lr_multiplier is not None:
            result["compression_lr_multiplier"] = self.compression_lr_multiplier
        return result
