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

from nncf import QuantizationPreset
from tests.post_training_quantization.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionHF
from tests.post_training_quantization.pipelines.base import ALL_NNCF_PTQ_BACKENDS
from tests.post_training_quantization.pipelines.base import BackendType
from tests.post_training_quantization.pipelines.image_classification_timm import ImageClassificationTimm
from tests.post_training_quantization.pipelines.masked_language_modeling import MaskedLanguageModelingHF

TEST_MODELS = [
    {
        "reported_name": "timm/resnet-18",
        "model_id": "resnet18",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {},
        "backends": ALL_NNCF_PTQ_BACKENDS,
        "params": {},
    },
    {
        "reported_name": "bert-base-uncased",
        "model_id": "bert-base-uncased",
        "pipeline_cls": MaskedLanguageModelingHF,
        "ptq_params": {"preset": QuantizationPreset.MIXED},
        "backends": [BackendType.OPTIMUM],
        "params": {},
    },
]


def generate_tests_scope():
    """
    Generate tests by names "{reported_name}_backend_{backend}"
    """
    tests_scope = {}
    for test_model_param in TEST_MODELS:
        for backend in test_model_param["backends"]:
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
