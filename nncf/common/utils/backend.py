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
from typing import Any, Callable, TypeVar, cast

from packaging import version

import nncf
from nncf.common.check_features import is_torch_tracing_by_patching

try:
    import openvino  # type: ignore # noqa: F401

    _OPENVINO_AVAILABLE = True
except ImportError:
    _OPENVINO_AVAILABLE = False

TModel = TypeVar("TModel")


class BackendType(Enum):
    TORCH = "Torch"
    TORCH_FX = "TorchFX"
    TENSORFLOW = "Tensorflow"
    ONNX = "ONNX"
    OPENVINO = "OpenVINO"


def result_verifier(func: Callable[[Any], bool]) -> Callable[..., None]:
    def verify_result(*args: Any, **kwargs: Any):  # type: ignore
        try:
            return func(*args, **kwargs)
        except Exception:
            return False

    return verify_result


@result_verifier
def is_torch_model(model: Any) -> bool:
    """
    Returns True if the model is an instance of torch.nn.Module and not a torch.fx.GraphModule, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of torch.nn.Module and not torch.fx.GraphModule, otherwise False.
    """
    import torch
    import torch.fx

    from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper

    if is_torch_tracing_by_patching():
        return not isinstance(model, torch.fx.GraphModule) and isinstance(model, torch.nn.Module)
    return isinstance(model, (GraphModelWrapper, torch.nn.Module)) and not isinstance(model, torch.fx.GraphModule)


@result_verifier
def is_torch_fx_model(model: Any) -> bool:
    """
    Returns True if the model is an instance of torch.fx.GraphModule, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of torch.fx.GraphModule, otherwise False.
    """
    import torch.fx

    return isinstance(model, torch.fx.GraphModule)


@result_verifier
def is_tensorflow_model(model: Any) -> bool:
    """
    Returns True if the model is an instance of tensorflow.Module, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of tensorflow.Module, otherwise False.
    """
    import tensorflow  # type: ignore

    return isinstance(model, tensorflow.Module)


@result_verifier
def is_onnx_model(model: Any) -> bool:
    """
    Returns True if the model is an instance of onnx.ModelProto, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of onnx.ModelProto, otherwise False.
    """
    import onnx  # type: ignore

    return isinstance(model, onnx.ModelProto)


@result_verifier
def is_openvino_model(model: Any) -> bool:
    """
    Returns True if the model is an instance of openvino.Model, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of openvino.Model, otherwise False.
    """
    import openvino as ov

    return isinstance(model, ov.Model)


@result_verifier
def is_openvino_compiled_model(model: Any) -> bool:
    """
    Returns True if the model is an instance of openvino.CompiledModel, otherwise False.

    :param model: A target model.
    :return: True if the model is an instance of openvino.CompiledModel, otherwise False.
    """
    import openvino as ov

    return isinstance(model, ov.CompiledModel)


def get_backend(model: Any) -> BackendType:
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

    msg = (
        "Could not infer the backend framework from the model type because "
        "the framework is not available or corrupted, or the model type is unsupported. "
    )
    raise nncf.UnsupportedBackendError(msg)


def copy_model(model: TModel) -> TModel:
    """
    Function to create copy of the backend-specific model.

    :param model: the backend-specific model instance
    :return: Copy of the backend-specific model instance.
    """
    model_backend = get_backend(model)
    if model_backend == BackendType.OPENVINO:
        # TODO(l-bat): Remove after fixing ticket: 100919
        from openvino import Model as OVModel

        ov_model = cast(OVModel, model)
        return cast(TModel, ov_model.clone())
    if model_backend == BackendType.TENSORFLOW:
        # deepcopy and tensorflow.keras.models.clone_model does not work correctly on 2.8.4 version
        from nncf.tensorflow.graph.model_transformer import TFModelTransformer
        from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout

        model = TFModelTransformer(model).transform(TFTransformationLayout())
        return model
    return deepcopy(model)


def is_openvino_available() -> bool:
    """
    Check if OpenVINO is available.

    :return: True if openvino package is installed, False otherwise.
    """
    return _OPENVINO_AVAILABLE


def is_openvino_at_least(version_str: str) -> bool:
    """
    Check if OpenVINO version is at least the specified one.

    :param version_str: The version string to compare with the installed OpenVINO version. For example "2025.1".
    :return: True if the installed OpenVINO version is at least the specified one, False otherwise.
    """
    if not _OPENVINO_AVAILABLE:
        return False

    openvino_version = version.parse(openvino.__version__.split("-")[0])
    return version.parse(version_str) <= openvino_version
