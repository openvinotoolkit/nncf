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

from typing import TypeVar
from abc import ABC, abstractmethod
from nncf.api.compression import CompressionAlgorithmController


ModelType = TypeVar('ModelType')

class TrainingRunner(ABC):

    @abstractmethod
    def train_epoch(self, model: ModelType, compression_controller: CompressionAlgorithmController):
        pass

    @abstractmethod
    def validate(self, model: ModelType, compression_controller: CompressionAlgorithmController):
        pass

    @abstractmethod
    def dump_checkpoint(self, model: ModelType, compression_controller: CompressionAlgorithmController):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def reset_training(self):
        pass

    @abstractmethod
    def retrieve_original_accuracy(self, model):
        pass
