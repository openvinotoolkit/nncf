"""
 Copyright (c) 2019-2020 Intel Corporation
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

from .version import __version__
from .common.utils.backend import __nncf_backend__

from .config import NNCFConfig

if __nncf_backend__ == 'Torch':
    # NNCF builds extensions based on torch load() function
    # This function has a bug inside which patch_extension_build_function() solves
    # This bug will be fixed in torch 1.8.0
    from .dynamic_graph.patch_pytorch import patch_extension_build_function
    # It must be called before importing packages containing CUDA extensions
    patch_extension_build_function()

    # Required for correct COMPRESSION_ALGORITHMS registry functioning
    from .binarization import algo as binarization_algo
    from .quantization import algo as quantization_algo
    from .sparsity.const import algo as const_sparsity_algo
    from .sparsity.magnitude import algo as magnitude_sparsity_algo
    from .sparsity.rb import algo as rb_sparsity_algo
    from .pruning.filter_pruning import algo as filter_pruning_algo

    # Functions most commonly used in integrating NNCF into training pipelines are
    # listed below for importing convenience

    from .model_creation import create_compressed_model
    from .checkpoint_loading import load_state
    from .common.utils.logger import disable_logging
    from .common.utils.logger import set_log_level
    from .initialization import register_default_init_args
    from .layers import register_module
    from .dynamic_graph.patch_pytorch import register_operator
    from .dynamic_graph.input_wrapping import nncf_model_input

    # NNCF relies on tracing PyTorch operations. Each code that uses NNCF
    # should be executed with PyTorch operators wrapped via a call to "patch_torch_operators",
    # so this call is moved to package __init__ to ensure this.
    from .dynamic_graph.patch_pytorch import patch_torch_operators

    from .extensions import force_build_cpu_extensions, force_build_cuda_extensions

    patch_torch_operators()
