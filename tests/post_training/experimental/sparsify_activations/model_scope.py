# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nncf.experimental.torch.sparsify_activations import TargetScope
from nncf.parameters import CompressWeightsMode
from tests.post_training.experimental.sparsify_activations.pipelines import ImageClassificationTimmSparsifyActivations
from tests.post_training.experimental.sparsify_activations.pipelines import LMSparsifyActivations
from tests.post_training.model_scope import generate_tests_scope
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
        "model_name": "tinyllama",
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
        "model_name": "tinyllama",
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
        "model_name": "timm/deit3_small_patch16_224",
        "pipeline_cls": ImageClassificationTimmSparsifyActivations,
        "compression_params": {},
        "backends": [BackendType.FP32],
        "batch_size": 128,
    },
    {
        "reported_name": "timm/deit3_small_patch16_224_qkv_sparse20_fc1_sparse20_fc2_sparse30",
        "model_id": "deit3_small_patch16_224",
        "model_name": "timm/deit3_small_patch16_224",
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


SPARSIFY_ACTIVATIONS_TEST_CASES = generate_tests_scope(SPARSIFY_ACTIVATIONS_MODELS)
