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
"""
Base subpackage for NNCF TensorFlow functionality.
"""

import tensorflow
from packaging import version
from packaging.specifiers import SpecifierSet

import nncf
from nncf import nncf_logger
from nncf.common.logging.logger import warn_bkc_version_mismatch
from nncf.version import BKC_TF_SPEC
from nncf.version import STRICT_TF_SPEC

try:
    tensorflow_version = version.parse(version.parse(tensorflow.__version__).base_version)
except:
    nncf_logger.debug("Could not parse tensorflow version")
    tensorflow_version = version.parse("0.0.0")

if tensorflow_version not in SpecifierSet(STRICT_TF_SPEC):
    raise nncf.UnsupportedVersionError(
        f"NNCF only supports tensorflow{STRICT_TF_SPEC}, while current tensorflow version is {tensorflow_version}"
    )
if tensorflow_version not in SpecifierSet(BKC_TF_SPEC):
    warn_bkc_version_mismatch("torch", BKC_TF_SPEC, tensorflow.__version__)


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
