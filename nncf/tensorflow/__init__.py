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

# pylint: skip-file
from nncf.version import BKC_TF_VERSION

import tensorflow
from pkg_resources import parse_version
if parse_version(BKC_TF_VERSION).base_version != parse_version(tensorflow.__version__).base_version:
    raise RuntimeError(
         'NNCF only supports tensorflow=={bkc}, while current tensorflow version is {curr}'.format(
         bkc=BKC_TF_VERSION,
         curr=tensorflow.__version__
    ))

from nncf.tensorflow.backend import backend

from nncf.tensorflow.helpers import create_compressed_model
from nncf.tensorflow.helpers.callback_creation import create_compression_callbacks
from nncf.tensorflow.initialization import register_default_init_args

# Required for correct COMPRESSION_ALGORITHMS registry functioning
from nncf.tensorflow.quantization import algorithm as quantization_algorithm
from nncf.tensorflow.sparsity.magnitude import algorithm as magnitude_sparsity_algorithm
from nncf.tensorflow.pruning.filter_pruning import algorithm as filter_pruning_algorithm
from nncf.tensorflow.sparsity.rb import algorithm as rb_sparsity_algorithm

from tensorflow.python.keras.engine import keras_tensor
keras_tensor.disable_keras_tensors()
