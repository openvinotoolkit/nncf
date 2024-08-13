# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TypeVar

import nncf
from nncf.common.engine import Engine
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.command_creation import CommandCreator
from nncf.common.tensor_statistics import aggregator
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data.dataset import Dataset

TModel = TypeVar("TModel")


class NNCFGraphFactory:
    @staticmethod
    def create(model: TModel) -> NNCFGraph:
        """
        Factory method to create backend-specific NNCFGraph instance based on the input model.

        :param model: backend-specific model instance
        :return: backend-specific NNCFGraph instance
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            from nncf.onnx.graph.nncf_graph_builder import GraphConverter

            return GraphConverter.create_nncf_graph(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.openvino.graph.nncf_graph_builder import GraphConverter

            return GraphConverter.create_nncf_graph(model)
        if model_backend == BackendType.TORCH_FX:
            from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter

            return GraphConverter.create_nncf_graph(model)
        if model_backend == BackendType.TORCH:
            return model.nncf.get_graph()
        raise nncf.UnsupportedBackendError(
            "Cannot create backend-specific graph because {} is not supported!".format(model_backend.value)
        )


class ModelTransformerFactory:
    @staticmethod
    def create(model: TModel, inplace: bool = False) -> ModelTransformer:
        """
        Factory method to create backend-specific ModelTransformer instance based on the input model.

        :param model: backend-specific model instance
        :param inplace: apply transformations inplace
        :return: backend-specific ModelTransformer instance
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            from nncf.onnx.graph.model_transformer import ONNXModelTransformer

            return ONNXModelTransformer(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.openvino.graph.model_transformer import OVModelTransformer

            return OVModelTransformer(model, inplace=inplace)
        if model_backend == BackendType.TORCH:
            from nncf.torch.model_transformer import PTModelTransformer

            return PTModelTransformer(model)
        if model_backend == BackendType.TORCH_FX:
            from nncf.experimental.torch.fx.model_transformer import FXModelTransformer

            return FXModelTransformer(model)
        raise nncf.UnsupportedBackendError(
            "Cannot create backend-specific model transformer because {} is not supported!".format(model_backend.value)
        )


class EngineFactory:
    @staticmethod
    def create(model: TModel) -> Engine:
        """
        Factory method to create backend-specific Engine instance based on the input model.

        :param model: backend-specific model instance.
        :return: backend-specific Engine instance.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            from nncf.onnx.engine import ONNXEngine

            return ONNXEngine(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.openvino.engine import OVNativeEngine

            return OVNativeEngine(model)
        if model_backend in (BackendType.TORCH, BackendType.TORCH_FX):
            from nncf.torch.engine import PTEngine

            return PTEngine(model)
        raise nncf.UnsupportedBackendError(
            "Cannot create backend-specific engine because {} is not supported!".format(model_backend.value)
        )


class CommandCreatorFactory:
    @staticmethod
    def create(model: TModel) -> CommandCreator:
        """
        Factory method to create backend-specific `CommandCreator` instance based on the input model.

        :param model: backend-specific model instance
        :return: backend-specific CommandCreator instance
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.openvino.graph.transformations.command_creation import OVCommandCreator

            return OVCommandCreator()

        if model_backend == BackendType.ONNX:
            from nncf.onnx.graph.transformations.command_creation import ONNXCommandCreator

            return ONNXCommandCreator()

        raise nncf.UnsupportedBackendError(
            "Cannot create backend-specific command creator because {} is not supported!".format(model_backend.value)
        )


class StatisticsAggregatorFactory:
    @staticmethod
    def create(model: TModel, dataset: Dataset) -> aggregator.StatisticsAggregator:
        """
        Factory method to create backend-specific `StatisticsAggregator` instance based on the input model.

        :param model: backend-specific model instance
        :return: backend-specific `StatisticsAggregator` instance
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            from nncf.onnx.statistics.aggregator import ONNXStatisticsAggregator

            return ONNXStatisticsAggregator(dataset)
        if model_backend == BackendType.OPENVINO:
            from nncf.openvino.statistics.aggregator import OVStatisticsAggregator

            return OVStatisticsAggregator(dataset)
        if model_backend == BackendType.TORCH:
            from nncf.torch.statistics.aggregator import PTStatisticsAggregator

            return PTStatisticsAggregator(dataset)
        if model_backend == BackendType.TORCH_FX:
            from nncf.experimental.torch.fx.statistics.aggregator import FXStatisticsAggregator

            return FXStatisticsAggregator(dataset)
        raise nncf.UnsupportedBackendError(
            "Cannot create backend-specific statistics aggregator because {} is not supported!".format(
                model_backend.value
            )
        )
