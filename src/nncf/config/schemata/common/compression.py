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
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import with_attributes

COMPRESSION_LR_MULTIPLIER_PROPERTY = {
    "compression_lr_multiplier": with_attributes(
        NUMBER,
        description="PyTorch only - Used to increase/decrease gradients "
        "for compression algorithms' parameters. The gradients will be multiplied "
        "by the specified value. If unspecified, the gradients will not be adjusted.",
    )
}
BASIC_COMPRESSION_ALGO_SCHEMA = {"type": "object", "required": ["algorithm"]}
