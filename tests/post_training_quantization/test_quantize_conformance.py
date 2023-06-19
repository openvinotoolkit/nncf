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
from pathlib import Path

import pandas as pd
import pytest
import transformers

from nncf import QuantizationPreset
from tests.post_training_quantization.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionHF
from tests.post_training_quantization.pipelines.base import ALL_NNCF_PTQ_BACKENDS
from tests.post_training_quantization.pipelines.base import BackendType
from tests.post_training_quantization.pipelines.image_classification import ImageClassificationHF
from tests.post_training_quantization.pipelines.masked_language_modeling import MaskedLanguageModelingHF

TEST_MODELS = [
    {
        "reported_name": "microsoft/resnet-18",
        "model_id": "microsoft/resnet-18",
        "pipeline_cls": ImageClassificationHF,
        "ptq_params": {"preset": QuantizationPreset.MIXED},
        "backends": ALL_NNCF_PTQ_BACKENDS,
        "params": {"pt_model_class": transformers.ResNetForImageClassification},
    },
    # {
    #     "reported_name": "openai/whisper-tiny",
    #     "model_id": "openai/whisper-tiny",
    #     "pipeline_cls": AutomaticSpeechRecognitionHF,
    #     "ptq_params": {"preset": QuantizationPreset.MIXED},
    #     "backends": ALL_NNCF_PTQ_BACKENDS,
    #     "params": {"pt_model_class": transformers.WhisperForConditionalGeneration},
    # },
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
    tests_scope = {}
    for test_model_param in TEST_MODELS:
        for backend in test_model_param["backends"]:
            model_param = copy.deepcopy(test_model_param)
            test_case_name = f"{model_param['reported_name']}_{backend.value}"
            model_param["backend"] = backend
            model_param.pop("backends")
            tests_scope[test_case_name] = model_param
    return tests_scope


TEST_CASES = generate_tests_scope()


@pytest.fixture(scope="session", name="cache_dir")
def fixture_data(pytestconfig):
    return pytestconfig.getoption("cache_dir")


@pytest.fixture(scope="session", name="output_dir")
def fixture_output(pytestconfig):
    return pytestconfig.getoption("output_dir")


@pytest.fixture(scope="session", name="result")
def fixture_result(pytestconfig):
    return pytestconfig.test_results


@pytest.mark.parametrize("test_case_name", TEST_CASES.keys())
def test_ptq_hf(test_case_name, output_dir, cache_dir, result):
    pipeline = None
    err_msg = None

    try:
        test_model_param = TEST_CASES[test_case_name]
        pipeline_cls = test_model_param["pipeline_cls"]

        pipeline_kwargs = {
            "reported_name": test_model_param["reported_name"],
            "model_id": test_model_param["model_id"],
            "backend": test_model_param["backend"],
            "ptq_params": test_model_param["ptq_params"],
            "num_samples": 1,
            "params": test_model_param["params"],
            "output_dir": Path(output_dir),
            "cache_dir": cache_dir,
        }

        pipeline = pipeline_cls(**pipeline_kwargs)

        pipeline.prepare()
        pipeline.quantize()

    except Exception as e:
        err_msg = str(e)
        raise Exception() from e
    finally:
        result_dict = {}
        if pipeline is not None:
            result_dict.update(pipeline.get_result_dict())
        result_dict["err_msg"] = err_msg
        result[test_case_name] = result_dict
