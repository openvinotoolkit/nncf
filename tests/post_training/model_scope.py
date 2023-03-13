"""
 Copyright (c) 2023 Intel Corporation
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
from typing import List

from nncf import QuantizationPreset
from nncf import ModelType

def get_validation_scope() -> List[dict]:
    model_scope = []
    # Basic
    model_scope.append({"name": "vgg11", "quantization_params": {}})
    model_scope.append({"name": "resnet18", "quantization_params": {}})
    model_scope.append({"name": "mobilenetv2_050", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    model_scope.append({"name": "mobilenetv2_050", "quantization_params": {"preset":QuantizationPreset.MIXED, "fast_bias_correction": False}})
    model_scope.append({"name": "repvgg_a2", "quantization_params": {}})
    # Densenet121 temporary excluded due to memory leaks
    # model_scope.append({"name": "densenet121", "quantization_params": {}})
    # model_scope.append({"name": "densenet121", "quantization_params": {"fast_bias_correction": False}})
    model_scope.append({"name": "tf_inception_v3", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    model_scope.append({"name": "xception", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    model_scope.append({"name": "efficientnet_b0", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    model_scope.append({"name": "efficientnet_b0", "quantization_params": {"preset":QuantizationPreset.MIXED, "fast_bias_correction": False}})
    model_scope.append({"name": "darknet53", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    # ResNets
    model_scope.append({"name": "seresnet18", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    model_scope.append({"name": "resnest14d", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    model_scope.append({"name": "inception_resnet_v2", "quantization_params": {}})
    model_scope.append({"name": "resnetv2_50d", "quantization_params": {}})
    model_scope.append({"name": "wide_resnet50_2", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    model_scope.append({"name": "bat_resnext26ts", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    model_scope.append({"name": "regnetx_002", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    # MobileNets
    model_scope.append({"name": "mobilenetv3_small_050", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    # Transformers
    model_scope.append({"name": "vit_small_patch16_18x2_224", "quantization_params": {"preset":QuantizationPreset.MIXED, "model_type": ModelType.TRANSFORMER}})
    model_scope.append({"name": "levit_128", "quantization_params": {"preset":QuantizationPreset.MIXED, "model_type": ModelType.TRANSFORMER}})
    model_scope.append({"name": "deit3_small_patch16_224", "quantization_params": {"preset":QuantizationPreset.MIXED, "model_type": ModelType.TRANSFORMER}})
    model_scope.append({"name": "swin_base_patch4_window7_224", "quantization_params": {"preset":QuantizationPreset.MIXED, "model_type": ModelType.TRANSFORMER}})
    model_scope.append({"name": "swinv2_cr_tiny_224", "quantization_params": {"preset":QuantizationPreset.MIXED, "model_type": ModelType.TRANSFORMER}})
    # convit_tiny supressed due to bug - 104173
    # model_scope.append({"name": "convit_tiny", "quantization_params": {"preset":QuantizationPreset.MIXED, "model_type": ModelType.TRANSFORMER}})
    model_scope.append({"name": "visformer_small", "quantization_params": {"preset":QuantizationPreset.MIXED, "model_type": ModelType.TRANSFORMER}})
    model_scope.append({"name": "crossvit_9_240", "quantization_params": {"preset":QuantizationPreset.MIXED, "model_type": ModelType.TRANSFORMER}})
    # Others
    model_scope.append({"name": "hrnet_w18", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    model_scope.append({"name": "efficientnet_lite0", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    model_scope.append({"name": "ghostnet_050", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    model_scope.append({"name": "dpn68", "quantization_params": {"preset":QuantizationPreset.MIXED}})
    model_scope.append({"name": "dla34", "quantization_params": {"preset":QuantizationPreset.MIXED}})

    return model_scope


VALIDATION_SCOPE = get_validation_scope()
