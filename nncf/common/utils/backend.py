# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
from copy import deepcopy
from enum import Enum
from typing import List, TypeVar

TModel = TypeVar("TModel")


class BackendType(Enum):
    TORCH = "Torch"
    TENSORFLOW = "Tensorflow"
    ONNX = "ONNX"
    OPENVINO = "OpenVINO"


def get_available_backends() -> List[BackendType]:
    """
    Returns a list of available backends.

    :return: A list of available backends.
    """
    frameworks = [
        ("torch", BackendType.TORCH),
        ("tensorflow", BackendType.TENSORFLOW),
        ("onnx", BackendType.ONNX),
        ("openvino.runtime", BackendType.OPENVINO),
    ]

    available_backends = []
    for module_name, backend in frameworks:
        try:
            importlib.import_module(module_name)
            available_backends.append(backend)
        except ImportError:
            pass

    return available_backends


def is_torch_model(model: TModel) -> bool:
    """
    Returns True if the model is an instance of torch.nn.Module, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of torch.nn.Module, otherwise False.
    """
    import torch

    return isinstance(model, torch.nn.Module)


def is_tensorflow_model(model: TModel) -> bool:
    """
    Returns True if the model is an instance of tensorflow.Module, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of tensorflow.Module, otherwise False.
    """
    import tensorflow

    return isinstance(model, tensorflow.Module)


def is_onnx_model(model: TModel) -> bool:
    """
    Returns True if the model is an instance of onnx.ModelProto, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of onnx.ModelProto, otherwise False.
    """
    import onnx

    return isinstance(model, onnx.ModelProto)


def is_openvino_model(model: TModel) -> bool:
    """
    Returns True if the model is an instance of openvino.runtime.Model, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of openvino.runtime.Model, otherwise False.
    """
    import openvino.runtime as ov

    return isinstance(model, ov.Model)


def is_openvino_compiled_model(model: TModel) -> bool:
    """
    Returns True if the model is an instance of openvino.runtime.CompiledModel, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of openvino.runtime.CompiledModel, otherwise False.
    """
    import openvino.runtime as ov

    return isinstance(model, ov.CompiledModel)


def get_backend(model: TModel) -> BackendType:
    """
    Returns the NNCF backend name string inferred from the type of the model object passed into this function.

    :param model: The framework-specific model.
    :return: A BackendType representing the correct NNCF backend to be used when working with the framework.
    """
    available_backends = get_available_backends()

    if BackendType.TORCH in available_backends and is_torch_model(model):
        return BackendType.TORCH

    if BackendType.TENSORFLOW in available_backends and is_tensorflow_model(model):
        return BackendType.TENSORFLOW

    if BackendType.ONNX in available_backends and is_onnx_model(model):
        return BackendType.ONNX

    if BackendType.OPENVINO in available_backends and is_openvino_model(model):
        return BackendType.OPENVINO

    raise RuntimeError(
        "Could not infer the backend framework from the model type because "
        "the framework is not available or the model type is unsupported. "
        "The available frameworks found: {}.".format(", ".join([b.value for b in available_backends]))
    )


def copy_model(model: TModel) -> TModel:
    """
    Function to create copy of the backend-specific model.

    :param model: the backend-specific model instance
    :return: Copy of the backend-specific model instance.
    """
    model_backend = get_backend(model)
    if model_backend == BackendType.OPENVINO:
        # TODO(l-bat): Remove after fixing ticket: 100919
        return model.clone()
    if model_backend == BackendType.TENSORFLOW:
        # deepcopy and tensorflow.keras.models.clone_model does not work correctly on 2.8.4 version
        from nncf.tensorflow.graph.model_transformer import TFModelTransformer
        from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout

        model = TFModelTransformer(model).transform(TFTransformationLayout())
        return model
    return deepcopy(model)
