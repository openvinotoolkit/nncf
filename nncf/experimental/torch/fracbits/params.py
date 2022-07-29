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

from dataclasses import dataclass, fields
from typing import Dict


class FracBitsParamsBase:
    @classmethod
    def from_config(cls, config: Dict) -> "FracBitsParamsBase":
        return cls(**{k: v for k, v in config.items() if k in fields(cls)})


@dataclass
class FracBitsLossParams(FracBitsParamsBase):
    type: str = "model_size"
    compression_rate: float = 1.5
    criteria: str = "L1"
    flip_loss: bool = False
    alpha: float = 10.0


@dataclass
class FracBitsSchedulerParams(FracBitsParamsBase):
    freeze_epoch: int = -1
