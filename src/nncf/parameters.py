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

from enum import Enum
from enum import auto
from typing import Any

from nncf.common.utils.api_marker import api


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]) -> Any:
        return name.lower()


@api(canonical_alias="nncf.TargetDevice")
class TargetDevice(StrEnum):
    """
    Target device architecture for compression.

    Compression will take into account the value of this parameter in order to obtain the best performance
    for this type of device.
    """

    ANY = "ANY"
    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    CPU_SPR = "CPU_SPR"


@api(canonical_alias="nncf.ModelType")
class ModelType(StrEnum):
    """
    Describes the model type the specificity of which will be taken into account during compression.

    :param TRANSFORMER: Transformer-based models
        (https://arxiv.org/pdf/1706.03762.pdf)
    """

    TRANSFORMER = "transformer"


@api(canonical_alias="nncf.DropType")
class DropType(StrEnum):
    """
    Describes the accuracy drop type, which determines how the accuracy drop between
    the original model and the compressed model is calculated.

    :param ABSOLUTE: The accuracy drop is calculated as the absolute drop with respect
        to the results of the original model.
    :param RELATIVE: The accuracy drop is calculated relative to the results of
        the original model.
    """

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


@api(canonical_alias="nncf.CompressWeightsMode")
class CompressWeightsMode(StrEnum):
    """
    Defines a mode for weight compression.
    :param INT8_SYM: Stands for 8-bit integer symmetric quantization of all weights.
        Weights are quantized symmetrically without zero point.
        https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md#symmetric-quantization
    :param INT8_ASYM: The same as INT8_SYM mode, but weights are quantized to a primary precision asymmetrically
        with a typical non-fixed zero point.
        https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md#asymmetric-quantization
    :param INT4_SYM: Stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
        Weights are quantized to a primary precision symmetrically without zero point.
        All embeddings and the last layer are always compressed to a backup precision, which is INT8_ASYM,
        by default. All others are quantized whether to 4-bit integer or to a backup precision depending on
        criteria and the given ratio.
        https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md#symmetric-quantization
    :param INT4_ASYM: The same as INT4_SYM mode, but weights are quantized to a primary precision asymmetrically
        with a typical non-fixed zero point.
        https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md#asymmetric-quantization
    :param NF4: The the same as INT4_SYM mode, but primary precision is NF4 data type without zero point.
    :param INT8: Mode is deprecated and will be removed in future releases. Please use `INT8_ASYM` instead.
    :param MXFP4: MX-compliant FP4 format with E2M1 values sharing group-level E8M0 scale. The size of group is 32.
    :param MXFP8_E4M3: MX-compliant FP8 format with E4M3 values sharing group-level E8M0 scale. The size of group is 32.
    :param FP8_E4M3: A FP8 format with E4M3 values sharing group-level fp16 scale.
    :param FP4: A FP4 format with E2M1 values sharing group-level fp16 scale.
    :param CODEBOOK: Codebook (LUT) quantization format.
    :param ADAPTIVE_CODEBOOK: Adaptive codebook (LUT) quantization format.
    :param CB4_F8E4M3: Codebook (LUT) format with 16 fixed fp8 values in E4M3 format.
    """

    INT8_SYM = "int8_sym"
    INT8_ASYM = "int8_asym"
    INT4_SYM = "int4_sym"
    INT4_ASYM = "int4_asym"
    NF4 = "nf4"
    CB4_F8E4M3 = "cb4_f8e4m3"
    INT8 = "int8"  # Deprecated mode
    MXFP4 = "mxfp4"
    MXFP8_E4M3 = "mxfp8_e4m3"
    FP8_E4M3 = "fp8_e4m3"
    FP4 = "fp4"
    CODEBOOK = "codebook"
    ADAPTIVE_CODEBOOK = "adaptive_codebook"


@api(canonical_alias="nncf.CompressionFormat")
class CompressionFormat(StrEnum):
    """
    Describes the format in which the model is saved after weight compression.

    :param DQ: Represents the 'dequantize' format, where weights are stored in low-bit precision,
        and a dequantization subgraph is added to the model. This is the default format for post-training weight
        compression methods.
    :param FQ: Represents the 'fake_quantize' format, where quantization is simulated by applying
        quantization and dequantization operations. Weights remain in the same precision. This format is
        suitable for quantization-aware training (QAT).
    :param FQ_LORA: Represents the 'fake_quantize_with_lora' format, which combines fake quantization
        with absorbable low-rank adapters (LoRA). Quantization is applied to the sum of weights and
        the multiplication of adapters. This makes quantization-aware training (QAT) more efficient in terms of
        accuracy, as adapters can also be tuned and remain computationally affordable during training due to their
        small dimensions.
    :param FQ_LORA_NLS: Represents the 'fake_quantize_with_lora_nls' format, which extends FQ_LORA with elastic
        absorbable low-rank adapters (LoRA). Quantization is applied similarly to FQ_LORA, and utilizing NLS often
        results in better performance for downstream task fine-tuning.
    """

    DQ = "dequantize"
    FQ = "fake_quantize"
    FQ_LORA = "fake_quantize_with_lora"
    FQ_LORA_NLS = "fake_quantize_with_lora_nls"


@api(canonical_alias="nncf.StripFormat")
class StripFormat(StrEnum):
    """
    Describes the format in which model is saved after strip: operation that removes auxiliary layers and
    operations added during the compression process, resulting in a clean model ready for deployment.
    The functionality of the model object is still preserved as a compressed model.

    :param NATIVE: Preserves as many custom NNCF additions as possible in the model.
    :param DQ: Replaces FakeQuantize operations with a dequantization subgraph and stores compressed weights
        in low-bit precision using fake quantize parameters. This is the default format for deploying models
        with compressed weights.
    :param IN_PLACE: Directly applies NNCF operations to the weights, replacing the original weights.
    """

    NATIVE = "native"
    DQ = "dequantize"
    IN_PLACE = "in_place"


@api(canonical_alias="nncf.BackupMode")
class BackupMode(StrEnum):
    """
    Defines a backup mode for weight compression.
    :param NONE: Stands for original floating-point precision of the model weights.
        In this mode, weights are retained in their original precision without any quantization.
    :param INT8_SYM: Stands for 8-bit integer symmetric quantization without zero point.
        https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md#symmetric-quantization
    :param INT8_ASYM: Stands for 8-bit integer asymmetric quantization with a typical non-fixed zero point.
        https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md#asymmetric-quantization
    """

    NONE = "none"
    INT8_SYM = "int8_sym"
    INT8_ASYM = "int8_asym"


@api(canonical_alias="nncf.SensitivityMetric")
class SensitivityMetric(StrEnum):
    """
    Defines a sensitivity metric for assigning quantization precision to layers. In order to
        preserve the accuracy of the model, the more sensitive layers receives a higher precision.

    :param WEIGHT_QUANTIZATION_ERROR: The inverted 8-bit quantization noise. Weights with highest value
        of this metric can be accurately quantized channel-wise to 8-bit. The idea is to leave these weights in 8bit,
        and quantize the rest of layers to 4-bit group-wise. Since group-wise is more accurate than per-channel,
        accuracy should not degrade.
    :param HESSIAN_INPUT_ACTIVATION: The average Hessian trace of weights with respect to the layer-wise quantization
        error multiplied by L2 norm of 8-bit quantization noise.
    :param MEAN_ACTIVATION_VARIANCE: The mean variance of the layers' inputs
        multiplied by inverted 8-bit quantization noise.
    :param MAX_ACTIVATION_VARIANCE: The maximum variance of the layers' inputs
        multiplied by inverted 8-bit quantization noise.
    :param MEAN_ACTIVATION_MAGNITUDE: The mean magnitude of the layers' inputs
        multiplied by inverted 8-bit quantization noise.
    """

    WEIGHT_QUANTIZATION_ERROR = "weight_quantization_error"
    HESSIAN_INPUT_ACTIVATION = "hessian_input_activation"
    MEAN_ACTIVATION_VARIANCE = "mean_activation_variance"
    MAX_ACTIVATION_VARIANCE = "max_activation_variance"
    MEAN_ACTIVATION_MAGNITUDE = "mean_activation_magnitude"


@api(canonical_alias="nncf.QuantizationMode")
class QuantizationMode(StrEnum):
    """
    Defines special modes.
    Currently contains only FP8-related modes (https://arxiv.org/pdf/2209.05433.pdf).

    :param FP8_E4M3: Mode with 4-bit exponent and 3-bit mantissa.
    :param FP8_E5M2: Mode with 5-bit exponent and 2-bit mantissa.
    """

    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


@api(canonical_alias="nncf.PruneMode")
class PruneMode(StrEnum):
    """
    Enumeration for different pruning modes in neural network compression.

    :param UNSTRUCTURED_MAGNITUDE_LOCAL: Unstructured magnitude-based pruning with local importance calculation.
        Weight importance is computed independently for each tensor.
    :param UNSTRUCTURED_MAGNITUDE_GLOBAL: Unstructured magnitude-based pruning with **global** importance calculation.
        Weight importance is computed across all tensors selected for pruning.
    :param UNSTRUCTURED_REGULARIZATION_BASED: Unstructured pruning based on trainable regularization masks.
        Trainable masks are introduced for the weights and optimized during training.
        This mode requires an additional regularization loss `RBLoss`.
    """

    UNSTRUCTURED_MAGNITUDE_LOCAL = auto()
    UNSTRUCTURED_MAGNITUDE_GLOBAL = auto()
    UNSTRUCTURED_REGULARIZATION_BASED = auto()
