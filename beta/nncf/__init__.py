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

from beta.nncf.version import __version__
from beta.nncf.backend import backend

from nncf.config import NNCFConfig

from beta.nncf import tensorflow as nncf_tensorflow
from beta.nncf.helpers.model_creation import create_compressed_model
from beta.nncf.helpers.callback_creation import create_compression_callbacks

from tensorflow.python.keras.engine import keras_tensor
keras_tensor.disable_keras_tensors()
