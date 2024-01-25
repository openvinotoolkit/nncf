# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Dict

from nncf.parameters import CompressWeightsMode
from tests.post_training.pipelines.base import BackendType
from tests.weight_compression.pipelines.lm_weight_compression import LMWeightCompression

BACKENDS = [BackendType.OV]
TEST_MODELS = [
    {
        "reported_name": "tinyllama_data_free",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "ptq_params": {
            "group_size": 64,
            "ratio": 0.8,
            "mode": CompressWeightsMode.INT4_SYM
        },
        "backends": BACKENDS,
    },
    # {
    #     "reported_name": "tinyllama/tinyllama-1.1b-step-50k-105b",
    #     "model_id": "tinyllama_data_aware",
    #     "pipeline_cls": LMWeightCompression,
    #     "cw_params": {
    #         "group_size": 64,
    #         "ratio": 0.8,
    #         "mode": CompressWeightsMode.INT4_SYM
    #     },
    #     "backends": BACKENDS,
    # },
]


def generate_tests_scope() -> Dict[str, dict]:
    """
    Generate tests by names "{reported_name}_backend_{backend}"
    """
    tests_scope = {}
    for test_model_param in TEST_MODELS:
        for backend in test_model_param["backends"] + [BackendType.FP32]:
            model_param = copy.deepcopy(test_model_param)
            reported_name = model_param["reported_name"]
            test_case_name = f"{reported_name}_backend_{backend.value}"
            model_param["backend"] = backend
            model_param.pop("backends")
            if test_case_name in tests_scope:
                raise RuntimeError(f"{test_case_name} already in tests_scope")
            tests_scope[test_case_name] = model_param
    return tests_scope


TEST_CASES = generate_tests_scope()
