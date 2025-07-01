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
from nncf.config.definitions import ONLINE_DOCS_ROOT
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.defaults import NUM_BN_ADAPTATION_SAMPLES

BATCHNORM_ADAPTATION_SCHEMA = {
    "type": "object",
    "description": f"This initializer is applied by default to utilize batch norm statistics adaptation to the "
    f"current compression scenario. See "
    f"[documentation]"
    f"({ONLINE_DOCS_ROOT}docs/compression_algorithms/Quantization.md#batch-norm-statistics-adaptation) "
    f"for more details.",
    "properties": {
        "num_bn_adaptation_samples": with_attributes(
            NUMBER,
            description="Number of samples from the training "
            "dataset to use for model inference "
            "during the BatchNorm statistics "
            "adaptation procedure for the compressed "
            "model. The actual number of samples will "
            "be a closest multiple of the batch "
            "size. Set this to 0 to disable BN adaptation.",
            default=NUM_BN_ADAPTATION_SAMPLES,
        )
    },
    "additionalProperties": False,
}
