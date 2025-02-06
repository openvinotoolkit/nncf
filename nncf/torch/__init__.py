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
# isort: off

"""
Base subpackage for NNCF PyTorch functionality.
"""

import os
from nncf import nncf_logger
from nncf.common.logging.logger import warn_bkc_version_mismatch

from nncf.version import BKC_TORCH_SPEC

import torch
from packaging import version
from packaging.specifiers import SpecifierSet

try:
    _torch_version = version.parse(version.parse(torch.__version__).base_version)
except:  # noqa: E722
    nncf_logger.debug("Could not parse torch version")
    _torch_version = version.parse("0.0.0")

if _torch_version not in SpecifierSet(BKC_TORCH_SPEC):
    warn_bkc_version_mismatch("torch", BKC_TORCH_SPEC, torch.__version__)


# Required for correct COMPRESSION_ALGORITHMS registry functioning
from nncf.torch.quantization import algo as quantization_algo
from nncf.torch.sparsity.const import algo as const_sparsity_algo
from nncf.torch.sparsity.magnitude import algo as magnitude_sparsity_algo
from nncf.torch.sparsity.rb import algo as rb_sparsity_algo
from nncf.experimental.torch.sparsity.movement import algo as movement_sparsity_algo
from nncf.torch.pruning.filter_pruning import algo as filter_pruning_algo
from nncf.torch.knowledge_distillation import algo as knowledge_distillation_algo

# Functions most commonly used in integrating NNCF into training pipelines are
# listed below for importing convenience

from nncf.torch.model_creation import create_compressed_model
from nncf.torch.model_creation import is_wrapped_model
from nncf.torch.model_creation import wrap_model
from nncf.torch.model_creation import load_from_config
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.initialization import register_default_init_args
from nncf.torch.layers import register_module
from nncf.torch.dynamic_graph.patch_pytorch import register_operator
from nncf.torch.dynamic_graph.io_handling import nncf_model_input
from nncf.torch.dynamic_graph.io_handling import nncf_model_output
from nncf.torch.dynamic_graph.context import disable_tracing
from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.dynamic_graph.context import forward_nncf_trace
from nncf.torch.strip import strip
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching

# NNCF relies on tracing PyTorch operations. Each code that uses NNCF
# should be executed with PyTorch operators wrapped via a call to "patch_torch_operators",
# so this call is moved to package __init__ to ensure this.
from nncf.torch.dynamic_graph.patch_pytorch import patch_torch_operators

from nncf.torch.extensions import force_build_cpu_extensions, force_build_cuda_extensions

# This is required since torchvision changes a dictionary inside of pytorch mapping
# different ops and their role in torch fx graph. Once the nncf mapping is done, it is
# represented as a different custom operation which is how it is changed in
# the said mapping. The polyfills loader is the specific file to be imported
# before making wrapping changes
if torch.__version__ >= "2.5.0":
    from torch._dynamo.polyfills import loader

if os.getenv("NNCF_EXPERIMENTAL_TORCH_TRACING") is None:
    patch_torch_operators()
