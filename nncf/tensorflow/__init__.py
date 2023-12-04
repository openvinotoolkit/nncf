# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base subpackage for NNCF TensorFlow functionality.
"""

import tensorflow
from packaging import version

from nncf import nncf_logger
from nncf.common.logging.logger import warn_bkc_version_mismatch
from nncf.version import BKC_TF_VERSION

try:
    _tf_version = tensorflow.__version__
    tensorflow_version = version.parse(_tf_version).base_version
except:
    nncf_logger.debug("Could not parse tensorflow version")
    _tf_version = "0.0.0"
    tensorflow_version = version.parse(_tf_version).base_version
tensorflow_version_major, tensorflow_version_minor = tuple(map(int, tensorflow_version.split(".")))[:2]
if not tensorflow_version.startswith(BKC_TF_VERSION[:-2]):
    warn_bkc_version_mismatch("tensorflow", BKC_TF_VERSION, _tf_version)
elif not (tensorflow_version_major == 2 and 8 <= tensorflow_version_minor <= 13):
    raise RuntimeError(
        f"NNCF only supports 2.8.4 <= tensorflow <= 2.13.*, while current tensorflow version is {_tf_version}"
    )


from nncf.common.accuracy_aware_training.training_loop import (
    AdaptiveCompressionTrainingLoop as AdaptiveCompressionTrainingLoop,
)
from nncf.common.accuracy_aware_training.training_loop import (
    EarlyExitCompressionTrainingLoop as EarlyExitCompressionTrainingLoop,
)
from nncf.tensorflow.helpers import create_compressed_model as create_compressed_model
from nncf.tensorflow.helpers.callback_creation import create_compression_callbacks as create_compression_callbacks
from nncf.tensorflow.initialization import register_default_init_args as register_default_init_args
from nncf.tensorflow.pruning.filter_pruning import algorithm as filter_pruning_algorithm

# Required for correct COMPRESSION_ALGORITHMS registry functioning
from nncf.tensorflow.quantization import algorithm as quantization_algorithm
from nncf.tensorflow.sparsity.magnitude import algorithm as magnitude_sparsity_algorithm
from nncf.tensorflow.sparsity.rb import algorithm as rb_sparsity_algorithm
