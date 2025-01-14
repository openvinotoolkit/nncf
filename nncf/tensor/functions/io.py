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

import functools
from pathlib import Path
from typing import Dict, Optional

from nncf.common.utils.os import fail_if_symlink
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDeviceType
from nncf.tensor.functions.dispatcher import dispatch_dict
from nncf.tensor.functions.dispatcher import get_io_backend_fn


def load_file(
    file_path: Path,
    *,
    backend: TensorBackend,
    device: Optional[TensorDeviceType] = None,
) -> Dict[str, Tensor]:
    """
    Loads a file containing tensor data and returns a dictionary of tensors.

    :param file_path: The path to the file to be loaded.
    :param backend: The backend type to determine the loading function.
    :param device: The device on which the tensor will be allocated, If device is not given,
        then the default device is determined by backend.
    :return: A dictionary where the keys are tensor names and the values are Tensor objects.
    """
    fail_if_symlink(file_path)
    loaded_dict = get_io_backend_fn("load_file", backend)(file_path, device=device)
    return {key: Tensor(val) for key, val in loaded_dict.items()}


@functools.singledispatch
def save_file(
    data: Dict[str, Tensor],
    file_path: Path,
) -> None:
    """
    Saves a dictionary of tensors to a file.

    :param data: A dictionary where the keys are tensor names and the values are Tensor objects.
    :param file_path: The path to the file where the tensor data will be saved.
    """
    fail_if_symlink(file_path)
    if isinstance(data, dict):
        return dispatch_dict(save_file, data, file_path)
    raise NotImplementedError(f"Function `save_file` is not implemented for {type(data)}")
