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

import tensorflow as tf
from safetensors.tensorflow import load_file as tf_load_file
from safetensors.tensorflow import save_file as tf_save_file

from nncf.tensor import TensorDeviceType
from nncf.tensor.functions import io as io
from nncf.tensor.functions.tf_numeric import DEVICE_MAP


def load_file(file_path: Path, *, device: Optional[TensorDeviceType] = None) -> dict[str, tf.Tensor]:
    loaded_tensors = tf_load_file(file_path)

    if device is not None:
        device_str = DEVICE_MAP[device]
        with tf.device(device_str):
            loaded_tensors = {k: tf.identity(v) for k, v in loaded_tensors.items()}

    return loaded_tensors


@io.save_file.register
def _(data: dict[str, tf.Tensor], file_path: Path) -> None:
    if file_path.is_symlink():
        from nncf.errors import ValidationError

        msg = "Cannot save tensor to a symbolic link"
        raise ValidationError(msg)

    return tf_save_file(data, file_path)
