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

from nncf import ModelType
from nncf import QuantizationPreset
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters
from tests.post_training.pipelines.base import ALL_PTQ_BACKENDS
from tests.post_training.pipelines.base import NNCF_PTQ_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.causal_language_model import CausalLMHF
from tests.post_training.pipelines.image_classification_timm import ImageClassificationTimm
from tests.post_training.pipelines.masked_language_modeling import MaskedLanguageModelingHF

TEST_MODELS = [
    # HF models
    {
        "reported_name": "hf/bert-base-uncased",
        "model_id": "bert-base-uncased",
        "pipeline_cls": MaskedLanguageModelingHF,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "subset_size": 2,
        },
        "backends": ALL_PTQ_BACKENDS + [BackendType.OPTIMUM],
    },
    {
        "reported_name": "hf/hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
        "model_id": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
        "pipeline_cls": CausalLMHF,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "subset_size": 2,
        },
        "backends": [BackendType.OPTIMUM],
    },
    # Timm models
    {
        "reported_name": "timm/crossvit_9_240",
        "model_id": "crossvit_9_240",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "advanced_parameters": AdvancedQuantizationParameters(smooth_quant_alpha=-1.0),
        },
        "backends": [BackendType.TORCH, BackendType.ONNX, BackendType.OV, BackendType.POT],
    },
    {
        "reported_name": "timm/darknet53",
        "model_id": "darknet53",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/deit3_small_patch16_224",
        "model_id": "deit3_small_patch16_224",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "advanced_parameters": AdvancedQuantizationParameters(
                smooth_quant_alphas=AdvancedSmoothQuantParameters(matmul=-1)
            ),
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/dla34",
        "model_id": "dla34",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/dpn68",
        "model_id": "dpn68",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/efficientnet_b0",
        "model_id": "efficientnet_b0",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/efficientnet_b0_BC",
        "model_id": "efficientnet_b0",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
            "fast_bias_correction": False,
        },
        "backends": [BackendType.ONNX, BackendType.OV, BackendType.POT],
    },
    {
        "reported_name": "timm/efficientnet_lite0",
        "model_id": "efficientnet_lite0",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/hrnet_w18",
        "model_id": "hrnet_w18",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/inception_resnet_v2",
        "model_id": "inception_resnet_v2",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {},
        "backends": NNCF_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/levit_128",
        "model_id": "levit_128",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
            "advanced_parameters": AdvancedQuantizationParameters(
                smooth_quant_alphas=AdvancedSmoothQuantParameters(matmul=0.05)
            ),
        },
        "backends": [BackendType.TORCH, BackendType.ONNX, BackendType.OV],
    },
    {
        "reported_name": "timm/mobilenetv2_050",
        "model_id": "mobilenetv2_050",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/mobilenetv2_050_BC",
        "model_id": "mobilenetv2_050",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
            "fast_bias_correction": False,
        },
        "backends": [BackendType.ONNX, BackendType.OV, BackendType.POT],
    },
    {
        "reported_name": "timm/mobilenetv3_small_050",
        "model_id": "mobilenetv3_small_050",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/regnetx_002",
        "model_id": "regnetx_002",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/resnest14d",
        "model_id": "resnest14d",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/resnet18",
        "model_id": "resnet18",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {},
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/swin_base_patch4_window7_224",
        "model_id": "swin_base_patch4_window7_224",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/tf_inception_v3",
        "model_id": "tf_inception_v3",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/vgg11",
        "model_id": "vgg11",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {},
        "backends": NNCF_PTQ_BACKENDS,
    },
    {
        "reported_name": "timm/visformer_small",
        "model_id": "visformer_small",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
            "model_type": ModelType.TRANSFORMER,
        },
        "backends": [BackendType.TORCH, BackendType.ONNX, BackendType.OV, BackendType.POT],
    },
    {
        "reported_name": "timm/wide_resnet50_2",
        "model_id": "wide_resnet50_2",
        "pipeline_cls": ImageClassificationTimm,
        "ptq_params": {
            "preset": QuantizationPreset.MIXED,
        },
        "backends": ALL_PTQ_BACKENDS,
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
