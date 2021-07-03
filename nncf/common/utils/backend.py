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
from enum import Enum

from nncf.api.compression import CompressionAlgorithmController


class BackendType(Enum):
    TORCH = 'Torch'
    TENSORFLOW = 'Tensorflow'


def infer_backend_from_model(model) -> BackendType:
    """
    Returns the NNCF backend name string inferred from the type of the model object passed into this function.

    :param model: The framework-specific object representing the trainable model.
    :return: A BackendType representing the correct NNCF backend to be used when working with the framework.
    """
    try:
        import torch
    except ImportError:
        torch = None

    try:
        import tensorflow
    except ImportError:
        tensorflow = None

    if torch is not None and isinstance(model, torch.nn.Module):
        return BackendType.TORCH

    if tensorflow is not None and isinstance(model, tensorflow.Module):
        return BackendType.TENSORFLOW

    raise RuntimeError('Could not infer the backend framework from the model type because '
                       'the framework is not available or the model type is unsupported.')


def infer_backend_from_compression_controller(compression_controller: CompressionAlgorithmController) -> BackendType:
    """
    Returns the NNCF backend name string inferred from the type of the model
    stored in the passed compression controller.

    :param compression_controller: Passed compression controller
    (of CompressionAlgorithmController type).
    :return: A BackendType representing the NNCF backend.
    """
    return infer_backend_from_model(compression_controller.model)
