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

from enum import Enum

from nncf.common.utils.api_marker import api


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


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
        https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#asymmetric-quantization
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
    :param E2M1: FP4 format from "OCP Microscaling Formats (MX) Specification" Version 1.0.
    """

    INT8_SYM = "int8_sym"
    INT8_ASYM = "int8_asym"
    INT4_SYM = "int4_sym"
    INT4_ASYM = "int4_asym"
    NF4 = "nf4"
    INT8 = "int8"  # Deprecated mode
    E2M1 = "e2m1"


@api(canonical_alias="nncf.BackupMode")
class BackupMode(StrEnum):
    """
    Defines a backup mode for weight compression.
    :param NONE: Stands for original floating-point precision of the model weights.
        In this mode, weights are retained in their original precision without any quantization.
    :param INT8_SYM: Stands for 8-bit integer symmetric quantization without zero point.
        https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md#symmetric-quantization
    :param INT8_ASYM: Stands for 8-bit integer asymmetric quantization with a typical non-fixed zero point.
        https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#asymmetric-quantization
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
