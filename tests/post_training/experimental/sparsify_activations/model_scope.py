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

import copy
from typing import Dict, List

import nncf
from nncf.experimental.torch.sparsify_activations import TargetScope
from nncf.parameters import CompressWeightsMode
from tests.post_training.experimental.sparsify_activations.pipelines import ImageClassificationTimmSparsifyActivations
from tests.post_training.experimental.sparsify_activations.pipelines import LMSparsifyActivations
from tests.post_training.pipelines.base import BackendType

SPARSIFY_ACTIVATIONS_MODELS = [
    {
        "reported_name": "tinyllama",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMSparsifyActivations,
        "compression_params": {},
        "backends": [BackendType.FP32],
    },
    {
        "reported_name": "tinyllama_ffn_sparse20",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMSparsifyActivations,
        "compression_params": {
            "compress_weights": None,
            "sparsify_activations": {
                "target_sparsity_by_scope": {
                    TargetScope(patterns=[".*up_proj.*", ".*gate_proj.*", ".*down_proj.*"]): 0.2,
                }
            },
        },
        "backends": [BackendType.TORCH, BackendType.CUDA_TORCH],
        "batch_size": 8,
    },
    {
        "reported_name": "tinyllama_int8_asym_data_free_ffn_sparse20",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMSparsifyActivations,
        "compression_params": {
            "compress_weights": {
                "mode": CompressWeightsMode.INT8_ASYM,
            },
            "sparsify_activations": {
                "target_sparsity_by_scope": {
                    TargetScope(patterns=[".*up_proj.*", ".*gate_proj.*", ".*down_proj.*"]): 0.2,
                }
            },
        },
        "backends": [BackendType.TORCH, BackendType.CUDA_TORCH],
        "batch_size": 8,
    },
    {
        "reported_name": "timm/deit3_small_patch16_224",
        "model_id": "deit3_small_patch16_224",
        "pipeline_cls": ImageClassificationTimmSparsifyActivations,
        "compression_params": {},
        "backends": [BackendType.FP32],
        "batch_size": 128,
    },
    {
        "reported_name": "timm/deit3_small_patch16_224_qkv_sparse20_fc1_sparse20_fc2_sparse30",
        "model_id": "deit3_small_patch16_224",
        "pipeline_cls": ImageClassificationTimmSparsifyActivations,
        "compression_params": {
            "sparsify_activations": {
                "target_sparsity_by_scope": {
                    TargetScope(patterns=[".*qkv.*", ".*fc1.*"]): 0.2,
                    TargetScope(patterns=[".*fc2.*"]): 0.3,
                }
            },
        },
        "backends": [BackendType.TORCH, BackendType.CUDA_TORCH],
        "batch_size": 128,
    },
]


def generate_tests_scope(models_list: List[Dict]) -> Dict[str, Dict]:
    """
    Generate tests by names "{reported_name}_backend_{backend}"
    """
    tests_scope = {}
    fp32_models = set()
    for test_model_param in models_list:
        model_id = test_model_param["model_id"]
        reported_name = test_model_param["reported_name"]

        for backend in test_model_param["backends"]:
            model_param = copy.deepcopy(test_model_param)
            if "is_batch_size_supported" not in model_param:  # Set default value of is_batch_size_supported.
                model_param["is_batch_size_supported"] = True
            test_case_name = f"{reported_name}_backend_{backend.value}"
            model_param["backend"] = backend
            model_param.pop("backends")
            if backend == BackendType.FP32:
                if model_id in fp32_models:
                    raise nncf.ValidationError(f"Duplicate test case for {model_id} with FP32 backend")
                fp32_models.add(model_id)
            if test_case_name in tests_scope:
                raise nncf.ValidationError(f"{test_case_name} already in tests_scope")
            tests_scope[test_case_name] = model_param

    return tests_scope


SPARSIFY_ACTIVATIONS_TEST_CASES = generate_tests_scope(SPARSIFY_ACTIVATIONS_MODELS)
