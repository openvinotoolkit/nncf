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

from abc import ABC
from abc import abstractmethod
from typing import Dict, Union

import nncf
from nncf.api.compression import CompressionAlgorithmController
from nncf.common.accuracy_aware_training.runner import BaseAccuracyAwareTrainingRunner
from nncf.common.accuracy_aware_training.runner import BaseAdaptiveCompressionLevelTrainingRunner
from nncf.common.accuracy_aware_training.runner import TrainingRunner
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend


class TrainingRunnerCreator(ABC):
    """
    Declares the factory method returning TrainingRunner object
    """

    @abstractmethod
    def create_training_loop(self) -> TrainingRunner:
        pass


class EarlyExitTrainingRunnerCreator(TrainingRunnerCreator):
    """
    Class creates an Early Exit Training Runner depending on an used backend.
    """

    def __init__(
        self,
        accuracy_aware_training_params: Dict[str, Union[float, int]],
        compression_controller: CompressionAlgorithmController,
        uncompressed_model_accuracy: float,
        verbose: bool,
        dump_checkpoints: bool,
        lr_updates_needed: bool,
    ):
        self.accuracy_aware_training_params = accuracy_aware_training_params
        self.compression_controller = compression_controller
        self.lr_updates_needed = lr_updates_needed
        self.verbose = verbose
        self.dump_checkpoints = dump_checkpoints
        self.uncompressed_model_accuracy = uncompressed_model_accuracy

    def create_training_loop(self) -> BaseAccuracyAwareTrainingRunner:
        """
        Creates an object of AccuracyAwareTrainingRunner depending on user backend

        :return: AccuracyAwareTrainingRunner object
        """
        nncf_backend = get_backend(self.compression_controller.model)  # type: ignore
        if nncf_backend is BackendType.TORCH:
            from nncf.torch.accuracy_aware_training.runner import PTAccuracyAwareTrainingRunner

            return PTAccuracyAwareTrainingRunner(
                self.accuracy_aware_training_params,
                self.uncompressed_model_accuracy,
                self.verbose,
                self.dump_checkpoints,
                self.lr_updates_needed,
            )
        if nncf_backend == BackendType.TENSORFLOW:
            from nncf.tensorflow.accuracy_aware_training.runner import TFAccuracyAwareTrainingRunner

            return TFAccuracyAwareTrainingRunner(
                self.accuracy_aware_training_params,
                self.uncompressed_model_accuracy,
                self.verbose,
                self.dump_checkpoints,
                self.lr_updates_needed,
            )
        raise nncf.UnsupportedBackendError("Got an unsupported value of nncf_backend")


class AdaptiveCompressionLevelTrainingRunnerCreator(TrainingRunnerCreator):
    """
    Class creates an Adaptive Compression Level Training Runner depending on an used backend.
    """

    def __init__(
        self,
        accuracy_aware_training_params: Dict[str, Union[float, int]],
        compression_controller: CompressionAlgorithmController,
        uncompressed_model_accuracy: float,
        verbose: bool,
        dump_checkpoints: bool,
        lr_updates_needed: bool,
        minimal_compression_rate: float,
        maximal_compression_rate: float,
    ):
        self.accuracy_aware_training_params = accuracy_aware_training_params
        self.compression_controller = compression_controller
        self.uncompressed_model_accuracy = uncompressed_model_accuracy
        self.lr_updates_needed = lr_updates_needed
        self.verbose = verbose
        self.minimal_compression_rate = minimal_compression_rate
        self.maximal_compression_rate = maximal_compression_rate
        self.dump_checkpoints = dump_checkpoints

    def create_training_loop(self) -> BaseAdaptiveCompressionLevelTrainingRunner:
        """
        Creates an object of AdaptiveCompressionLevelTrainingRunner depending on user backend

        :return: AdaptiveCompressionLevelTrainingRunner object
        """
        nncf_backend = get_backend(self.compression_controller.model)  # type: ignore

        if nncf_backend is BackendType.TORCH:
            from nncf.torch.accuracy_aware_training.runner import PTAdaptiveCompressionLevelTrainingRunner

            return PTAdaptiveCompressionLevelTrainingRunner(
                self.accuracy_aware_training_params,
                self.uncompressed_model_accuracy,
                self.verbose,
                self.dump_checkpoints,
                self.lr_updates_needed,
                self.minimal_compression_rate,
                self.maximal_compression_rate,
            )
        if nncf_backend == BackendType.TENSORFLOW:
            from nncf.tensorflow.accuracy_aware_training.runner import TFAdaptiveCompressionLevelTrainingRunner

            return TFAdaptiveCompressionLevelTrainingRunner(
                self.accuracy_aware_training_params,
                self.uncompressed_model_accuracy,
                self.verbose,
                self.dump_checkpoints,
                self.lr_updates_needed,
                self.minimal_compression_rate,
                self.maximal_compression_rate,
            )
        raise nncf.UnsupportedBackendError("Got an unsupported value of nncf_backend")
