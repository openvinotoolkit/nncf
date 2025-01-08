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
from typing import Dict, Type

import tensorflow as tf

from nncf.api.compression import CompressionAlgorithmController
from nncf.api.compression import CompressionStage
from nncf.common.compression import NO_COMPRESSION_ALGORITHM_NAME
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.backend import copy_model
from nncf.common.utils.registry import Registry
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.tensorflow.loss import TFZeroCompressionLoss

TF_COMPRESSION_ALGORITHMS = Registry("compression algorithm", add_name_as_attr=True)


@TF_COMPRESSION_ALGORITHMS.register(NO_COMPRESSION_ALGORITHM_NAME)
class NoCompressionAlgorithmBuilder(TFCompressionAlgorithmBuilder):
    def get_transformation_layout(self, _) -> TransformationLayout:
        return TransformationLayout()

    def _build_controller(self, model: tf.keras.Model) -> CompressionAlgorithmController:
        return NoCompressionAlgorithmController(model)

    def initialize(self, model: tf.keras.Model) -> None:
        pass

    def _get_algo_specific_config_section(self) -> Dict:
        return {}


class NoCompressionAlgorithmController(BaseCompressionAlgorithmController):
    def __init__(self, target_model: tf.keras.Model):
        super().__init__(target_model)
        self._loss = TFZeroCompressionLoss()
        self._scheduler = StubCompressionScheduler()

    @property
    def loss(self) -> TFZeroCompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> StubCompressionScheduler:
        return self._scheduler

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

    def strip(self, do_copy: bool = True) -> tf.keras.Model:
        model = self.model
        if do_copy:
            model = copy_model(self.model)
        return model

    def compression_stage(self) -> CompressionStage:
        return CompressionStage.UNCOMPRESSED


def get_compression_algorithm_builder(algo_name: str) -> Type[TFCompressionAlgorithmBuilder]:
    nncf_logger.info(f"Creating compression algorithm: {algo_name}")
    return TF_COMPRESSION_ALGORITHMS.get(algo_name)
