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
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, TypeVar

import nncf

TModel = TypeVar("TModel")


class BackendType(Enum):
    TORCH = "Torch"
    TORCH_FX = "TorchFX"
    TENSORFLOW = "Tensorflow"
    ONNX = "ONNX"
    OPENVINO = "OpenVINO"


def result_verifier(func: Callable[[TModel], bool]) -> Callable[..., None]:
    def verify_result(*args: Any, **kwargs: Any):  # type: ignore
        try:
            return func(*args, **kwargs)
        except Exception:
            return False

    return verify_result


@result_verifier
def is_torch_model(model: TModel) -> bool:
    """
    Returns True if the model is an instance of torch.nn.Module and not a torch.fx.GraphModule, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of torch.nn.Module and not torch.fx.GraphModule, otherwise False.
    """
    import torch
    import torch.fx

    return not isinstance(model, torch.fx.GraphModule) and isinstance(model, torch.nn.Module)


@result_verifier
def is_torch_fx_model(model: TModel) -> bool:
    """
    Returns True if the model is an instance of torch.fx.GraphModule, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of torch.fx.GraphModule, otherwise False.
    """
    import torch.fx

    return isinstance(model, torch.fx.GraphModule)


@result_verifier
def is_tensorflow_model(model: TModel) -> bool:
    """
    Returns True if the model is an instance of tensorflow.Module, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of tensorflow.Module, otherwise False.
    """
    import tensorflow  # type: ignore

    return isinstance(model, tensorflow.Module)


@result_verifier
def is_onnx_model(model: TModel) -> bool:
    """
    Returns True if the model is an instance of onnx.ModelProto, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of onnx.ModelProto, otherwise False.
    """
    import onnx  # type: ignore

    return isinstance(model, onnx.ModelProto)


@result_verifier
def is_openvino_model(model: TModel) -> bool:
    """
    Returns True if the model is an instance of openvino.runtime.Model, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of openvino.runtime.Model, otherwise False.
    """
    import openvino.runtime as ov  # type: ignore

    return isinstance(model, ov.Model)


@result_verifier
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

    verify_map = {
        is_torch_fx_model: BackendType.TORCH_FX,
        is_torch_model: BackendType.TORCH,
        is_tensorflow_model: BackendType.TENSORFLOW,
        is_onnx_model: BackendType.ONNX,
        is_openvino_model: BackendType.OPENVINO,
    }

    for backend_call, backend in verify_map.items():
        if backend_call(model):
            return backend

    raise nncf.UnsupportedBackendError(
        "Could not infer the backend framework from the model type because "
        "the framework is not available or corrupted, or the model type is unsupported. "
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
        return model.clone()  # type: ignore
    if model_backend == BackendType.TENSORFLOW:
        # deepcopy and tensorflow.keras.models.clone_model does not work correctly on 2.8.4 version
        from nncf.tensorflow.graph.model_transformer import TFModelTransformer
        from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout

        model = TFModelTransformer(model).transform(TFTransformationLayout())
        return model
    return deepcopy(model)
