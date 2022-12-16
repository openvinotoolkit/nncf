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
from typing import Any, Dict, List, Optional


class SchedulerParams:
    def __init__(self, power: Optional[int] = 3,
                 warmup_start_epoch: Optional[int] = 1,
                 warmup_end_epoch: Optional[int] = 3,
                 init_importance_threshold: Optional[float] = -1.0,
                 final_importance_threshold: Optional[float] = 0.0,
                 importance_regularization_factor: Optional[float] = 0.1,
                 steps_per_epoch: Optional[int] = 4,
                 enable_structured_masking: Optional[bool] = True):
        self.power = power
        self.warmup_start_epoch = warmup_start_epoch
        self.warmup_end_epoch = warmup_end_epoch
        self.init_importance_threshold = init_importance_threshold
        self.final_importance_threshold = final_importance_threshold
        self.importance_regularization_factor = importance_regularization_factor
        self.steps_per_epoch = steps_per_epoch
        self.enable_structured_masking = enable_structured_masking

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


class NNCFAlgoConfig:
    def __init__(self, sparse_structure_by_scopes: Optional[List[Dict]] = None,
                 ignored_scopes: Optional[List[str]] = None,
                 compression_lr_multiplier: Optional[float] = None,
                 scheduler_params: Optional[SchedulerParams] = None,
                 **scheduler_overrides):
        self.scheduler_params = scheduler_params or SchedulerParams()
        for k, v in scheduler_overrides.items():
            assert hasattr(self.scheduler_params, k)
            setattr(self.scheduler_params, k, v)
        self.sparse_structure_by_scopes = sparse_structure_by_scopes or []
        self.ignored_scopes = ignored_scopes or []
        self.compression_lr_multiplier = compression_lr_multiplier

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'algorithm': 'movement_sparsity',
            'params': self.scheduler_params.to_dict(),
            'sparse_structure_by_scopes': self.sparse_structure_by_scopes,
            'ignored_scopes': self.ignored_scopes,
        }
        if self.compression_lr_multiplier is not None:
            result['compression_lr_multiplier'] = self.compression_lr_multiplier
        return result
