"""
 Copyright (c) 2021 Intel Corporation
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

from enum import Enum
from nncf.config import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.config.extractors import extract_accuracy_aware_training_config
from nncf.common.accuracy_aware_training.training_loop import TrainingLoop
from nncf.common.accuracy_aware_training.training_loop import EarlyExitCompressionTrainingLoop
from nncf.common.accuracy_aware_training.training_loop import AdaptiveCompressionTrainingLoop


class AccuracyAwareTrainingMode(Enum):
    early_exit = 1
    adaptive_compression_level = 2


def create_accuracy_aware_training_loop(nncf_config: NNCFConfig,
                                        compression_ctrl: CompressionAlgorithmController) -> TrainingLoop:
    accuracy_aware_training_config = extract_accuracy_aware_training_config(nncf_config)
    accuracy_aware_training_mode = accuracy_aware_training_config.mode
    if accuracy_aware_training_mode == AccuracyAwareTrainingMode.early_exit:
        return EarlyExitCompressionTrainingLoop(nncf_config, compression_ctrl)
    elif accuracy_aware_training_mode == AccuracyAwareTrainingMode.adaptive_compression_level:
        return AdaptiveCompressionTrainingLoop(nncf_config, compression_ctrl)
