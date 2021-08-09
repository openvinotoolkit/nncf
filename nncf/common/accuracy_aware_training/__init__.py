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

from nncf.config import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.config.extractors import extract_accuracy_aware_training_params
from nncf.common.accuracy_aware_training.training_loop import TrainingLoop
from nncf.common.accuracy_aware_training.training_loop import EarlyExitCompressionTrainingLoop
from nncf.common.accuracy_aware_training.training_loop import AdaptiveCompressionTrainingLoop


class AccuracyAwareTrainingMode:
    EARLY_EXIT = 'early_exit'
    ADAPTIVE_COMPRESSION_LEVEL = 'adaptive_compression_level'


def create_accuracy_aware_training_loop(nncf_config: NNCFConfig,
                                        compression_ctrl: CompressionAlgorithmController,
                                        **kwargs) -> TrainingLoop:
    """
    Creates an accuracy aware training loop corresponding to NNCFConfig and CompressionAlgorithmController.
    :param: nncf_config: An instance of the NNCFConfig.
    :compression_ctrl: An instance of thr CompressionAlgorithmController.
    :return: Accuracy aware training loop.
    """
    accuracy_aware_training_params = extract_accuracy_aware_training_params(nncf_config)
    accuracy_aware_training_mode = accuracy_aware_training_params.get('mode')
    if accuracy_aware_training_mode == AccuracyAwareTrainingMode.EARLY_EXIT:
        return EarlyExitCompressionTrainingLoop(nncf_config, compression_ctrl, **kwargs)
    if accuracy_aware_training_mode == AccuracyAwareTrainingMode.ADAPTIVE_COMPRESSION_LEVEL:
        return AdaptiveCompressionTrainingLoop(nncf_config, compression_ctrl, **kwargs)
    raise RuntimeError('Incorrect accuracy aware mode in the config file')
