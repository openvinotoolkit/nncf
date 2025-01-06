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

from nncf import NNCFConfig
from nncf.config.utils import is_experimental_quantization


def patch_if_experimental_quantization(nncf_config: NNCFConfig):
    if "compression" in nncf_config and is_experimental_quantization(nncf_config):
        from nncf.experimental.tensorflow.patch_tf import patch_tf_operations

        patch_tf_operations()
