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

from typing import Dict, Optional

import numpy as np
from safetensors.numpy import load_file as np_load_file
from safetensors.numpy import save_file as np_save_file

from nncf.tensor.definitions import TensorDeviceType
from nncf.tensor.functions import io as io
from nncf.tensor.functions.dispatcher import register_numpy_types
from nncf.tensor.functions.numpy_numeric import validate_device


def load_file(file_path: str, *, device: Optional[TensorDeviceType] = None) -> Dict[str, np.ndarray]:
    validate_device(device)
    return np_load_file(file_path)


@register_numpy_types(io.save_file)
def _(data: Dict[str, np.ndarray], file_path: str) -> None:
    return np_save_file(data, file_path)
