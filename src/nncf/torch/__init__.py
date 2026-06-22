# Copyright (c) 2026 Intel Corporation
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

import torch

# Functions most commonly used in integrating NNCF into training pipelines are
# listed below for importing convenience

from nncf.torch.model_creation import is_wrapped_model as is_wrapped_model
from nncf.torch.model_creation import wrap_model as wrap_model
from nncf.torch.function_hook.serialization import load_from_config as load_from_config
from nncf.torch.function_hook.serialization import get_config as get_config
from nncf.torch.function_hook.hook_executor_mode import disable_tracing as disable_tracing
from nncf.torch.strip import strip as strip

# NNCF relies on tracing PyTorch operations. Each code that uses NNCF
# should be executed with PyTorch operators wrapped via a call to "patch_torch_operators",
# so this call is moved to package __init__ to ensure this.

from nncf.torch.extensions import force_build_cpu_extensions as force_build_cpu_extensions
from nncf.torch.extensions import force_build_cuda_extensions as force_build_cuda_extensions
