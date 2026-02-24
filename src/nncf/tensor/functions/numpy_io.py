# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from safetensors.numpy import load_file as np_load_file
from safetensors.numpy import save_file as np_save_file

from nncf.common.utils.os import fail_if_symlink
from nncf.tensor.definitions import TensorDeviceType
from nncf.tensor.functions import io as io
from nncf.tensor.functions.numpy_numeric import validate_device

T_NUMPY_ARRAY = NDArray[Any]
T_NUMPY = Union[T_NUMPY_ARRAY, np.generic]  # type: ignore[type-arg]


def load_file(file_path: str, *, device: Optional[TensorDeviceType] = None) -> dict[str, T_NUMPY_ARRAY]:
    validate_device(device)
    return np_load_file(file_path)


@io.save_file.register
def _(data: dict[str, T_NUMPY], file_path: Path) -> None:
    fail_if_symlink(file_path)
    np_save_file(data, file_path)  # type: ignore [arg-type]
