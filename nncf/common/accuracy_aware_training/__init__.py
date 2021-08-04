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
    EARLY_EXIT = 'early_exit'
    ADAPTIVE_COMPRESSION_LEVEL = 'adaptive_compression_level'

    @staticmethod
    def from_str(accuracy_aware_training_mode: str):
        if accuracy_aware_training_mode == AccuracyAwareTrainingMode.EARLY_EXIT.value:
            return EarlyExitCompressionTrainingLoop
        if accuracy_aware_training_mode == AccuracyAwareTrainingMode.ADAPTIVE_COMPRESSION_LEVEL.value:
            return  AdaptiveCompressionTrainingLoop
        raise RuntimeError('Incorrect accuracy_aware_training mode')


def create_accuracy_aware_training_loop(nncf_config: NNCFConfig,
                                        compression_ctrl: CompressionAlgorithmController,
                                        **kwargs) -> TrainingLoop:
    accuracy_aware_training_config = extract_accuracy_aware_training_config(nncf_config)
    accuracy_aware_training_mode = accuracy_aware_training_config.get('mode', None)
    training_loop_class = AccuracyAwareTrainingMode.from_str(accuracy_aware_training_mode)
    return training_loop_class(nncf_config, compression_ctrl, **kwargs)
