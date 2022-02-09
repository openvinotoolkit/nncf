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

from typing import TypeVar

from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.dataloader import DataLoader

from nncf.experimental.post_training.backend import BACKEND

ModelType = TypeVar('ModelType')


class CompressedModel:
    """
    The original model wrapper used to build NNCFGraph and utilized it in the compression algorithms.
    """

    def __init__(self, model: ModelType):
        self.original_model = model
        self._determine_backend(self.original_model)
        self.nncf_graph = None
        self.compressed_model = None  # Final compressed model. This model will be exported.
        self.transformed_model = None  # Model with applied transformtaions
        self.transformations = []

    def _determine_backend(self, model: ModelType) -> None:
        from torch.nn import Module
        from tensorflow.keras.models import Model
        from onnx import ModelProto
        if isinstance(model, ModelProto):
            self.model_backend = BACKEND.ONNX
            return
        elif isinstance(model, Module):
            self.model_backend = BACKEND.PYTORCH
            return
        elif isinstance(model, Model):
            self.model_backend = BACKEND.TENSORFLOW
            return
        elif isinstance(model, ):  # TODO: add OpenVINO
            self.model_backend = BACKEND.OPENVINO
            return
        raise RuntimeError('This backend is not supported')

    def build_and_set_nncf_graph(self, dataloader: DataLoader, engine: Engine):
        """
        Builds NNCFGraph from the model.
        """
        if self.model_backend == BACKEND.ONNX:
            from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter
            self.nncf_graph = GraphConverter.create_nncf_graph(self.original_model)

    def get_original_model(self) -> ModelType:
        return self.original_model
