"""
 Copyright (c) 2021 Intel Corporation
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

__nncf_backend__ = 'Torch'


def infer_backend_from_compression_controller(compression_controller):
    """
    Returns the NNCF backend name string inferred from the type of the model
    stored in the passed compression controller.

    :param compression_controller: Passed compression controller
    (of CompressionAlgorithmController type).
    :return: A string representing the NNCF backend name (either `Torch` or `TensorFlow`).
    """
    try:
        import torch
    except ImportError:
        torch = None

    try:
        import tensorflow
    except ImportError:
        tensorflow = None

    if torch is not None and isinstance(compression_controller.model, torch.nn.Module):
        return 'Torch'

    if tensorflow is not None and isinstance(compression_controller.model, tensorflow.Module):
        return 'TensorFlow'

    raise RuntimeError('Could not infer the backend framework from the model type because '
                       'the framework is not available or the model type is unsupported.')
