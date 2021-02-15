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

from typing import Optional, TypeVar

from nncf.api.compression import CompressionAlgorithmController
from nncf.api.compression import CompressionAlgorithmBuilder
from nncf.api.compression import CompressionScheduler
from beta.nncf.tensorflow.graph.model_transformer import TFModelTransformer
from beta.nncf.tensorflow.utils.save import save_model

ModelType = TypeVar('ModelType')
DatasetType = TypeVar('DatasetType')
LossType = TypeVar('LossType')


class TFCompressionScheduler(CompressionScheduler):
    def __init__(self):
        super().__init__()
        self.last_epoch = -1
        self.last_step = -1

    def step(self, last: Optional[int] = None) -> None:
        if last is None:
            last = self.last_step + 1
        self.last_step = last

    def epoch_step(self, last: Optional[int] = None) -> None:
        if last is None:
            last = self.last_epoch + 1
        self.last_epoch = last

    def load_state(self, initial_step: int, steps_per_epoch: int) -> None:
        self.last_step = initial_step - 1
        self.last_epoch = self.last_step // steps_per_epoch


class TFCompressionAlgorithmInitializer:
    def call(self,
             model: ModelType,
             dataset: Optional[DatasetType] = None,
             loss: Optional[LossType] = None) -> None:
        pass

    def __call__(self, *args, **kwargs) -> None:
        self.call(*args, **kwargs)


class TFCompressionAlgorithmController(CompressionAlgorithmController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as
    compression scheduler and compression loss.
    """

    def __init__(self, target_model: ModelType):
        """
        Initializes the internal state of the compression algorithm controller.

        :param target_model: The model with additional modifications necessary
            to enable algorithm-specific compression during fine-tuning built
            by the `CompressionAlgorithmBuilder`.
        """
        super().__init__(target_model)
        self._initializer = TFCompressionAlgorithmInitializer()
        self._scheduler = TFCompressionScheduler()

    def initialize(self,
                   dataset: Optional[DatasetType] = None,
                   loss: Optional[LossType] = None) -> None:
        self._initializer(self._model, dataset, loss)

    def export_model(self, save_path: str, save_format: str = 'frozen_graph') -> None:
        """
        Used to export the compressed model to the Frozen Graph, TensorFlow SavedModel,
        or Keras H5 formats. Makes method-specific preparations of the model, (e.g.
        removing auxiliary layers that were used for the model compression), then
        exports the model in the specified path.

        :param save_path: The path to export model.
        :param save_format: Saving format (`frozen_graph` for Frozen Graph,
            `tf` for Tensorflow SavedModel, `h5` for Keras H5 format).
        """
        self.prepare_for_export()
        save_model(self.model, save_path, save_format)


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
        return TFModelTransformer(model, transformation_layout).transform()

    def build_controller(self, model: ModelType) -> TFCompressionAlgorithmController:
        """
        Builds `CompressionAlgorithmController` to handle the additional modules,
        parameters, and hooks inserted into the model to enable algorithm-specific
        compression.

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        :return: The instance of the `CompressionAlgorithmController`.
        """
        return TFCompressionAlgorithmController(model)
