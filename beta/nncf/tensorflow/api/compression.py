"""
 Copyright (c) 2020 Intel Corporation
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
from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from nncf.api.compression import CompressionAlgorithmBuilder
from nncf.common.compression import BaseCompressionAlgorithmController
from beta.nncf.tensorflow.graph.model_transformer import TFModelTransformer

ModelType = TypeVar('ModelType')
DatasetType = TypeVar('DatasetType')
LossType = TypeVar('LossType')


class TFCompressionAlgorithmInitializer(ABC):
    @abstractmethod
    def call(self,
             model: ModelType,
             dataset: Optional[DatasetType] = None,
             loss: Optional[LossType] = None) -> None:
        """
        Initializes minimum and maximum quantization ranges.
        """

    def __call__(self, *args, **kwargs) -> None:
        self.call(*args, **kwargs)


class TFCompressionAlgorithmController(BaseCompressionAlgorithmController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as
    compression scheduler and compression loss.
    """

    def initialize(self,
                   dataset: Optional[DatasetType] = None,
                   loss: Optional[LossType] = None) -> None:
        pass


class TFCompressionAlgorithmBuilder(CompressionAlgorithmBuilder):
    """
    Determines which modifications should be made to the original model in
    order to enable algorithm-specific compression during fine-tuning.
    """

    def apply_to(self, model: ModelType) -> ModelType:
        """
        Applies algorithm-specific modifications to the model.

        :param model: The original uncompressed model.
        :return: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """
        transformation_layout = self.get_transformation_layout(model)
        transformer = TFModelTransformer(model, transformation_layout)
        transformed_model = transformer.transform()
        return transformed_model
