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

from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file as pt_load_file
from safetensors.torch import save_file as pt_save_file

from nncf.common.utils.os import fail_if_symlink
from nncf.tensor import TensorDeviceType
from nncf.tensor.functions import io as io
from nncf.tensor.functions.torch_numeric import convert_to_torch_device


def load_file(file_path: str, *, device: Optional[TensorDeviceType] = None) -> dict[str, torch.Tensor]:
    pt_device = convert_to_torch_device(device)
    if pt_device is None:
        pt_device = "cpu"
    return pt_load_file(file_path, device=pt_device)


@io.save_file.register
def _(data: dict[str, torch.Tensor], file_path: Path) -> None:
    fail_if_symlink(file_path)
    pt_save_file(data, file_path)
