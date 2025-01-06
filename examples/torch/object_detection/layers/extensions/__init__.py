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

import os.path

import torch

from nncf.torch.extensions import CudaNotAvailableStub
from nncf.torch.extensions import get_build_directory_for_extension

ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if torch.cuda.is_available():
    EXTENSIONS = torch.utils.cpp_extension.load(
        "extensions",
        [
            os.path.join(ext_dir, "extensions.cpp"),
            os.path.join(ext_dir, "nms/nms.cpp"),
            os.path.join(ext_dir, "nms/nms_kernel.cu"),
        ],
        verbose=False,
        build_directory=get_build_directory_for_extension("extensions"),
    )
else:
    EXTENSIONS = CudaNotAvailableStub
