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
from nncf import ModelType
from nncf import QuantizationPreset
from nncf.parameters import BackupMode
from nncf.parameters import CompressWeightsMode
from nncf.parameters import SensitivityMetric
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.advanced_parameters import AdvancedLoraCorrectionParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import AdvancedScaleEstimationParameters
from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters
from tests.post_training.pipelines.base import ALL_PTQ_BACKENDS
from tests.post_training.pipelines.base import NNCF_PTQ_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.causal_language_model import CausalLMHF
from tests.post_training.pipelines.gpt import GPT
from tests.post_training.pipelines.image_classification_timm import ImageClassificationTimm
from tests.post_training.pipelines.image_classification_torchvision import ImageClassificationTorchvision
from tests.post_training.pipelines.lm_weight_compression import LMWeightCompression
from tests.post_training.pipelines.masked_language_modeling import MaskedLanguageModelingHF

QUANTIZATION_MODELS = [
    # HF models
    {
        "reported_name": "hf/bert-base-uncased",
        "model_id": "bert-base-uncased",
        "pipeline_cls": MaskedLanguageModelingHF,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "subset_size": 2,
        },
        "backends": ALL_PTQ_BACKENDS + [BackendType.OPTIMUM],
    },
    {
        "reported_name": "hf/hf-internal-testing/tiny-random-GPTNeoXForCausalLM_statefull",
        "model_id": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
        "pipeline_cls": CausalLMHF,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "subset_size": 2,
        },
        "params": {"is_stateful": True},
        "backends": [BackendType.OPTIMUM],
    },
    {
        "reported_name": "hf/hf-internal-testing/tiny-random-GPTNeoXForCausalLM_stateless",
        "model_id": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
        "pipeline_cls": CausalLMHF,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "subset_size": 2,
        },
        "params": {"is_stateful": False},
        "backends": [BackendType.OPTIMUM],
    },
    {
        "reported_name": "hf/hf-internal-testing/tiny-random-gpt2",
        "model_id": "hf-internal-testing/tiny-random-gpt2",
        "pipeline_cls": GPT,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "subset_size": 2,
        },
        "backends": [BackendType.TORCH, BackendType.OV, BackendType.OPTIMUM],
    },
    # Torchvision models
    {
        "reported_name": "torchvision/resnet18",
        "model_id": "resnet18",
        "pipeline_cls": ImageClassificationTorchvision,
        "compression_params": {},
        "backends": [BackendType.FX_TORCH, BackendType.TORCH, BackendType.CUDA_TORCH, BackendType.OV, BackendType.ONNX],
        "batch_size": 128,
    },
    {
        "reported_name": "torchvision/mobilenet_v3_small_BC",
        "model_id": "mobilenet_v3_small",
        "pipeline_cls": ImageClassificationTorchvision,
        "compression_params": {
            "fast_bias_correction": False,
            "preset": QuantizationPreset.MIXED,
        },
        "backends": [BackendType.FX_TORCH, BackendType.OV, BackendType.ONNX],
        "batch_size": 128,
    },
    {
        "reported_name": "torchvision/vit_b_16",
        "model_id": "vit_b_16",
        "pipeline_cls": ImageClassificationTorchvision,
        "compression_params": {
            "model_type": ModelType.TRANSFORMER,
            "advanced_parameters": AdvancedQuantizationParameters(smooth_quant_alpha=0.15),
        },
        "backends": [BackendType.FX_TORCH, BackendType.OV],
        "batch_size": 1,
    },
    {
        "reported_name": "torchvision/swin_v2_s",
        "model_id": "swin_v2_s",
        "pipeline_cls": ImageClassificationTorchvision,
        "compression_params": {
            "model_type": ModelType.TRANSFORMER,
            "advanced_parameters": AdvancedQuantizationParameters(smooth_quant_alpha=0.5),
        },
        "backends": [BackendType.FX_TORCH, BackendType.OV],
        "batch_size": 1,
    },
    # Timm models
    {
        "reported_name": "timm/crossvit_9_240",
        "model_id": "crossvit_9_240",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "advanced_parameters": AdvancedQuantizationParameters(smooth_quant_alpha=-1.0),
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/darknet53",
        "model_id": "darknet53",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/deit3_small_patch16_224",
        "model_id": "deit3_small_patch16_224",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "advanced_parameters": AdvancedQuantizationParameters(
                smooth_quant_alphas=AdvancedSmoothQuantParameters(matmul=-1)
            ),
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/dla34",
        "model_id": "dla34",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/dpn68",
        "model_id": "dpn68",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/efficientnet_b0",
        "model_id": "efficientnet_b0",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/efficientnet_b0_BC",
        "model_id": "efficientnet_b0",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "fast_bias_correction": False,
        },
        "backends": [BackendType.ONNX, BackendType.OV],
        "batch_size": 128,
    },
    {
        "reported_name": "timm/efficientnet_lite0",
        "model_id": "efficientnet_lite0",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/hrnet_w18",
        "model_id": "hrnet_w18",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/inception_resnet_v2",
        "model_id": "inception_resnet_v2",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {},
        "backends": NNCF_PTQ_BACKENDS,
        "batch_size": 64,
    },
    {
        "reported_name": "timm/levit_128",
        "model_id": "levit_128",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "advanced_parameters": AdvancedQuantizationParameters(
                smooth_quant_alphas=AdvancedSmoothQuantParameters(matmul=0.05)
            ),
        },
        "backends": NNCF_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/mobilenetv2_050",
        "model_id": "mobilenetv2_050",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/mobilenetv2_050_BC",
        "model_id": "mobilenetv2_050",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "fast_bias_correction": False,
        },
        "backends": [BackendType.ONNX, BackendType.OV],
        "batch_size": 128,
    },
    {
        "reported_name": "timm/mobilenetv3_small_050",
        "model_id": "mobilenetv3_small_050",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/mobilenetv3_small_050_BC",
        "model_id": "mobilenetv3_small_050",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "fast_bias_correction": False,
        },
        "backends": [BackendType.ONNX, BackendType.OV],
        "batch_size": 128,
    },
    {
        "reported_name": "timm/regnetx_002",
        "model_id": "regnetx_002",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/resnest14d",
        "model_id": "resnest14d",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/swin_base_patch4_window7_224",
        "model_id": "swin_base_patch4_window7_224",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
        },
        "backends": [BackendType.OV],
        "batch_size": 32,
    },
    {
        "reported_name": "timm/swin_base_patch4_window7_224_no_sq",
        "model_id": "swin_base_patch4_window7_224",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "advanced_parameters": AdvancedQuantizationParameters(
                smooth_quant_alphas=AdvancedSmoothQuantParameters(matmul=-1)
            ),
        },
        "backends": [BackendType.TORCH, BackendType.CUDA_TORCH, BackendType.ONNX],
        "batch_size": 128,
    },
    {
        "reported_name": "timm/tf_inception_v3",
        "model_id": "tf_inception_v3",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/vgg11",
        "model_id": "vgg11",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {},
        "backends": NNCF_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/visformer_small",
        "model_id": "visformer_small",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
    {
        "reported_name": "timm/wide_resnet50_2",
        "model_id": "wide_resnet50_2",
        "pipeline_cls": ImageClassificationTimm,
        "compression_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
        "batch_size": 128,
    },
]


WEIGHT_COMPRESSION_MODELS = [
    {
        "reported_name": "tinyllama_data_free",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "compression_params": {
            "group_size": 64,
            "ratio": 0.8,
            "mode": CompressWeightsMode.INT4_SYM,
            "sensitivity_metric": SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
        },
        "backends": [BackendType.OV],
    },
    {
        "reported_name": "tinyllama_data_aware",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "compression_params": {"group_size": 64, "ratio": 0.8, "mode": CompressWeightsMode.INT4_SYM},
        "backends": [BackendType.OV],
    },
    {
        "reported_name": "tinyllama_data_aware_awq_stateful",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "compression_params": {"group_size": 64, "ratio": 0.8, "mode": CompressWeightsMode.INT4_SYM, "awq": True},
        "params": {"is_stateful": True},
        "backends": [BackendType.OV],
    },
    {
        "reported_name": "tinyllama_data_aware_awq_scale_estimation",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "compression_params": {
            "group_size": 64,
            "ratio": 0.8,
            "mode": CompressWeightsMode.INT4_SYM,
            "awq": True,
            "scale_estimation": True,
            "advanced_parameters": AdvancedCompressionParameters(
                scale_estimation_params=AdvancedScaleEstimationParameters(32, 5, 10, 1.0)
            ),
        },
        "backends": [BackendType.OV],
    },
    {
        "reported_name": "tinyllama_data_aware_awq_scale_estimation_stateful",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "compression_params": {
            "group_size": 64,
            "ratio": 0.8,
            "mode": CompressWeightsMode.INT4_SYM,
            "awq": True,
            "scale_estimation": True,
            "advanced_parameters": AdvancedCompressionParameters(
                scale_estimation_params=AdvancedScaleEstimationParameters(32, 5, 10, 1.0)
            ),
        },
        "params": {"is_stateful": True},
        "backends": [BackendType.OV],
    },
    {
        "reported_name": "tinyllama_int8_data_free",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "compression_params": {
            "mode": CompressWeightsMode.INT8_ASYM,
        },
        "backends": [BackendType.TORCH],
    },
    {
        "reported_name": "tinyllama_int4_data_free",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "compression_params": {
            "group_size": 64,
            "ratio": 0.8,
            "mode": CompressWeightsMode.INT4_SYM,
            "sensitivity_metric": SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
        },
        "backends": [BackendType.TORCH],
    },
    {
        "reported_name": "tinyllama_data_aware_gptq_scale_estimation_stateful",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "compression_params": {
            "group_size": 64,
            "ratio": 0.8,
            "mode": CompressWeightsMode.INT4_SYM,
            "gptq": True,
            "scale_estimation": True,
            "advanced_parameters": AdvancedCompressionParameters(
                scale_estimation_params=AdvancedScaleEstimationParameters(32, 5, 10, 1.0)
            ),
        },
        "params": {"is_stateful": True},
        "backends": [BackendType.OV],
    },
    {
        "reported_name": "tinyllama_NF4_scale_estimation_stateful_per_channel",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "compression_params": {
            "group_size": -1,
            "ratio": 0.1,
            "mode": CompressWeightsMode.NF4,
            "scale_estimation": True,
            "advanced_parameters": AdvancedCompressionParameters(
                scale_estimation_params=AdvancedScaleEstimationParameters(32, 5, 10, 1.0)
            ),
        },
        "params": {"is_stateful": True},
        "backends": [BackendType.OV],
    },
    {
        "reported_name": "tinyllama_scale_estimation_per_channel",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "compression_params": {
            "group_size": -1,
            "ratio": 0.8,
            "mode": CompressWeightsMode.INT4_ASYM,
            "scale_estimation": True,
        },
        "backends": [BackendType.OV],
    },
    {
        "reported_name": "tinyllama_data_aware_lora_stateful",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "compression_params": {
            "group_size": 64,
            "ratio": 0.8,
            "mode": CompressWeightsMode.INT4_SYM,
            "lora_correction": True,
            "advanced_parameters": AdvancedCompressionParameters(
                lora_correction_params=AdvancedLoraCorrectionParameters(
                    adapter_rank=8, num_iterations=3, apply_regularization=False, subset_size=32, use_int8_adapters=True
                )
            ),
        },
        "params": {"is_stateful": True},
        "backends": [BackendType.OV],
    },
    {
        "reported_name": "tinyllama_awq_backup_mode_none",
        "model_id": "tinyllama/tinyllama-1.1b-step-50k-105b",
        "pipeline_cls": LMWeightCompression,
        "compression_params": {
            "group_size": 64,
            "ratio": 0.8,
            "all_layers": True,
            "backup_mode": BackupMode.NONE,
            "mode": CompressWeightsMode.INT4_ASYM,
            "awq": True,
            "ignored_scope": nncf.IgnoredScope(types=["Gather"]),
        },
        "backends": [BackendType.OV],
    },
]


def generate_tests_scope(models_list: List[Dict]) -> Dict[str, dict]:
    """
    Generate tests by names "{reported_name}_backend_{backend}"
    """
    reported_name_to_model_id_mapping = {mc["reported_name"]: mc["model_id"] for mc in models_list}
    tests_scope = {}
    fp32_models = set()
    for test_model_param in models_list:
        for backend in test_model_param["backends"] + [BackendType.FP32]:
            model_param = copy.deepcopy(test_model_param)
            if "is_batch_size_supported" not in model_param:  # Set default value of is_batch_size_supported.
                model_param["is_batch_size_supported"] = True
            reported_name = model_param["reported_name"]
            model_id = reported_name_to_model_id_mapping[reported_name]
            if backend == BackendType.FP32:
                # Some test cases may share the same model_id, therefore fp32 test case is added only once for model_id.
                if model_id not in fp32_models:
                    fp32_models.add(model_id)
                else:
                    continue
            test_case_name = f"{reported_name}_backend_{backend.value}"
            model_param["backend"] = backend
            model_param.pop("backends")
            if test_case_name in tests_scope:
                raise nncf.ValidationError(f"{test_case_name} already in tests_scope")
            tests_scope[test_case_name] = model_param
    return tests_scope


PTQ_TEST_CASES = generate_tests_scope(QUANTIZATION_MODELS)
WC_TEST_CASES = generate_tests_scope(WEIGHT_COMPRESSION_MODELS)
