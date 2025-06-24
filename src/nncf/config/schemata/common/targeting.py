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
from nncf.config.schemata.basic import BOOLEAN
from nncf.config.schemata.basic import make_string_or_array_of_strings_schema
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.common.initialization import BATCHNORM_ADAPTATION_SCHEMA
from nncf.config.schemata.defaults import VALIDATE_SCOPES

IGNORED_SCOPES_DESCRIPTION = (
    "A list of model control flow graph node scopes to be ignored for this "
    "operation - functions as an 'allowlist'. Optional."
)
TARGET_SCOPES_DESCRIPTION = (
    "A list of model control flow graph node scopes to be considered for this operation"
    " - functions as a 'denylist'. Optional."
)
VALIDATE_SCOPES_DESCRIPTION = (
    "If set to True, then a RuntimeError will be raised if the names of the "
    "ignored/target scopes do not match the names of the scopes in the model graph."
)
SCOPING_PROPERTIES = {
    "ignored_scopes": with_attributes(
        make_string_or_array_of_strings_schema(),
        description=IGNORED_SCOPES_DESCRIPTION,
        examples=["{re}conv.*", ["LeNet/relu_0", "LeNet/relu_1"]],
    ),
    "target_scopes": with_attributes(
        make_string_or_array_of_strings_schema(),
        description=TARGET_SCOPES_DESCRIPTION,
        examples=[
            [
                "UNet/ModuleList[down_path]/UNetConvBlock[1]/Sequential[block]/Conv2d[0]",
                "UNet/ModuleList[down_path]/UNetConvBlock[2]/Sequential[block]/Conv2d[0]",
                "UNet/ModuleList[down_path]/UNetConvBlock[3]/Sequential[block]/Conv2d[0]",
                "UNet/ModuleList[down_path]/UNetConvBlock[4]/Sequential[block]/Conv2d[0]",
            ],
            "UNet/ModuleList\\[up_path\\].*",
        ],
    ),
    "validate_scopes": with_attributes(
        BOOLEAN,
        description=VALIDATE_SCOPES_DESCRIPTION,
        default=VALIDATE_SCOPES,
    ),
}
GENERIC_INITIALIZER_SCHEMA = {
    "type": "object",
    "properties": {"batchnorm_adaptation": BATCHNORM_ADAPTATION_SCHEMA},
    "additionalProperties": False,
}
