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

import torch
from safetensors.torch import load_file as pt_load_file
from safetensors.torch import save_file as pt_save_file

from nncf.tensor import TensorDeviceType
from nncf.tensor.functions import io as io
from nncf.tensor.functions.torch_numeric import convert_to_torch_device


def load_file(file_path: str, *, device: Optional[TensorDeviceType] = None) -> Dict[str, torch.Tensor]:
    device = convert_to_torch_device(device)
    return pt_load_file(file_path, device=device)


@io.save_file.register(torch.Tensor)
def _(data: Dict[str, torch.Tensor], file_path: str) -> None:
    return pt_save_file(data, file_path)
