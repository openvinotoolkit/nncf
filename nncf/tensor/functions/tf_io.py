# Copyright (c) 2024 Intel Corporation
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

import tensorflow as tf
from safetensors.tensorflow import load_file as tf_load_file
from safetensors.tensorflow import save_file as tf_save_file

from nncf.tensor import TensorDeviceType
from nncf.tensor.functions import io as io


def load_file(file_path: str, *, device: Optional[TensorDeviceType] = None) -> Dict[str, tf.Tensor]:
    return tf_load_file(file_path)


@io.save_file.register(tf.Tensor)
def _(data: Dict[str, tf.Tensor], file_path: str) -> None:
    return tf_save_file(data, file_path)
