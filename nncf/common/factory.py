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

from typing import Any, TypeVar, cast

import nncf
from nncf.common.check_features import is_torch_tracing_by_patching
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
            from onnx import ModelProto  # type: ignore

            from nncf.onnx.graph.nncf_graph_builder import GraphConverter as ONNXGraphConverter

            return ONNXGraphConverter.create_nncf_graph(cast(ModelProto, model))
        if model_backend == BackendType.OPENVINO:
            from openvino import Model  # type: ignore

            from nncf.openvino.graph.nncf_graph_builder import GraphConverter as OVGraphConverter

            return OVGraphConverter.create_nncf_graph(cast(Model, model))
        if model_backend == BackendType.TORCH_FX:
            from torch.fx import GraphModule

            from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter as FXGraphConverter

            return FXGraphConverter.create_nncf_graph(cast(GraphModule, model))
        if model_backend == BackendType.TORCH:
            from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper
            from nncf.torch.nncf_network import NNCFNetwork

            if isinstance(model, GraphModelWrapper):
                return model.get_graph()
            if isinstance(model, NNCFNetwork):
                return model.nncf.get_graph()
            msg = f"Unexpected type of model {type(model)} for TORCH backend"
            raise nncf.InternalError(msg)
        msg = f"Cannot create backend-specific graph because {model_backend.value} is not supported!"
        raise nncf.UnsupportedBackendError(msg)


class ModelTransformerFactory:
    @staticmethod
    def create(model: TModel, inplace: bool = False) -> ModelTransformer[Any]:
        """
        Factory method to create backend-specific ModelTransformer instance based on the input model.

        :param model: backend-specific model instance
        :param inplace: apply transformations inplace
        :return: backend-specific ModelTransformer instance
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            from onnx import ModelProto

            from nncf.onnx.graph.model_transformer import ONNXModelTransformer

            return ONNXModelTransformer(cast(ModelProto, model))
        if model_backend == BackendType.OPENVINO:
            from openvino import Model

            from nncf.openvino.graph.model_transformer import OVModelTransformer

            return OVModelTransformer(cast(Model, model), inplace=inplace)
        if model_backend == BackendType.TORCH and not is_torch_tracing_by_patching():
            from nncf.torch.function_hook.model_transformer import PT2ModelTransformer
            from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper

            return PT2ModelTransformer(cast(GraphModelWrapper, model))

        if model_backend == BackendType.TORCH and is_torch_tracing_by_patching():
            from nncf.torch.model_transformer import PTModelTransformer
            from nncf.torch.nncf_network import NNCFNetwork

            return PTModelTransformer(cast(NNCFNetwork, model))

        if model_backend == BackendType.TORCH_FX:
            from torch.fx import GraphModule

            from nncf.experimental.torch.fx.model_transformer import FXModelTransformer

            return FXModelTransformer(cast(GraphModule, model))
        msg = f"Cannot create backend-specific model transformer because {model_backend.value} is not supported!"
        raise nncf.UnsupportedBackendError(msg)


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
            from onnx import ModelProto

            from nncf.onnx.engine import ONNXEngine

            return ONNXEngine(cast(ModelProto, model))
        if model_backend == BackendType.OPENVINO:
            from openvino import Model

            from nncf.openvino.engine import OVNativeEngine

            return OVNativeEngine(cast(Model, model))
        if model_backend in (BackendType.TORCH, BackendType.TORCH_FX):
            from torch.nn import Module

            from nncf.torch.engine import PTEngine
            from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper

            if isinstance(model, GraphModelWrapper):
                pt_model = model.model
            else:
                pt_model = cast(Module, model)
            return PTEngine(pt_model)
        msg = f"Cannot create backend-specific engine because {model_backend.value} is not supported!"
        raise nncf.UnsupportedBackendError(msg)


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

        msg = f"Cannot create backend-specific command creator because {model_backend.value} is not supported!"
        raise nncf.UnsupportedBackendError(msg)


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
        if model_backend == BackendType.TORCH and is_torch_tracing_by_patching():
            from nncf.torch.statistics.aggregator import PTStatisticsAggregator

            return PTStatisticsAggregator(dataset)
        if model_backend == BackendType.TORCH and not is_torch_tracing_by_patching():
            from nncf.torch.function_hook.statistics.aggregator import PT2StatisticsAggregator

            return PT2StatisticsAggregator(dataset)
        if model_backend == BackendType.TORCH_FX:
            from nncf.experimental.torch.fx.statistics.aggregator import FXStatisticsAggregator

            return FXStatisticsAggregator(dataset)
        msg = f"Cannot create backend-specific statistics aggregator because {model_backend.value} is not supported!"
        raise nncf.UnsupportedBackendError(msg)
