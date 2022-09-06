"""
 Copyright (c) 2022 Intel Corporation
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
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import make_string_or_array_of_strings_schema
from nncf.config.schemata.basic import with_attributes

IGNORED_SCOPES_DESCRIPTION = "A list of model control flow graph node scopes to be ignored for this " \
                             "operation - functions as a 'allowlist'. Optional."
TARGET_SCOPES_DESCRIPTION = "A list of model control flow graph node scopes to be considered for this operation" \
                            " - functions as a 'denylist'. Optional."
SCOPING_PROPERTIES = {
    "ignored_scopes": with_attributes(make_string_or_array_of_strings_schema(),
                                      description=IGNORED_SCOPES_DESCRIPTION),
    "target_scopes": with_attributes(make_string_or_array_of_strings_schema(),
                                     description=TARGET_SCOPES_DESCRIPTION),
}
BATCHNORM_ADAPTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "num_bn_adaptation_samples": with_attributes(NUMBER,
                                                     description="Number of samples from the training "
                                                                 "dataset to use for model inference "
                                                                 "during the BatchNorm statistics "
                                                                 "adaptation procedure for the compressed "
                                                                 "model. The actual number of samples will "
                                                                 "be a closest multiple of the batch "
                                                                 "size.")
    },
    "additionalProperties": False,
}
GENERIC_INITIALIZER_SCHEMA = {
    "type": "object",
    "properties": {
        "batchnorm_adaptation": BATCHNORM_ADAPTATION_SCHEMA
    },
    "additionalProperties": False,
}
