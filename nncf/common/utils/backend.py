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
from enum import Enum

from nncf.api.compression import CompressionAlgorithmController


class BackendType(Enum):
    TORCH = 'Torch'
    TENSORFLOW = 'Tensorflow'
    ONNX = 'ONNX'
    OPENVINO = 'OpenVINO'


def get_backend(model) -> BackendType:
    """
    Returns the NNCF backend name string inferred from the type of the model object passed into this function.

    :param model: The framework-specific object representing the trainable model.
    :return: A BackendType representing the correct NNCF backend to be used when working with the framework.
    """
    available_frameworks = []
    try:
        import torch
        available_frameworks.append('PyTorch')
    except ImportError:
        torch = None

    try:
        import tensorflow
        available_frameworks.append('Tensorflow')
    except ImportError:
        tensorflow = None

    try:
        import onnx
        available_frameworks.append('ONNX')
    except ImportError:
        onnx = None

    try:
        import openvino.runtime as ov
        available_frameworks.append('OpenVINO')
    except ImportError:
        ov = None

    if torch is not None and isinstance(model, torch.nn.Module):
        return BackendType.TORCH

    if tensorflow is not None and isinstance(model, tensorflow.Module):
        return BackendType.TENSORFLOW

    if onnx is not None and isinstance(model, onnx.ModelProto):
        return BackendType.ONNX

    if ov is not None and isinstance(model, ov.Model):
        return BackendType.OPENVINO

    raise RuntimeError('Could not infer the backend framework from the model type because '
                       'the framework is not available or the model type is unsupported. '
                       'The available frameworks found: {}.'.format(', '.join(available_frameworks)))


def infer_backend_from_compression_controller(compression_controller: CompressionAlgorithmController) -> BackendType:
    """
    Returns the NNCF backend name string inferred from the type of the model
    stored in the passed compression controller.

    :param compression_controller: Passed compression controller
    (of CompressionAlgorithmController type).
    :return: A BackendType representing the NNCF backend.
    """
    return get_backend(compression_controller.model)
