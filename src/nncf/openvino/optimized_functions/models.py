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
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from typing import Callable, Optional, Union

import numpy as np
import openvino as ov
from openvino import Node
from openvino import opset13 as opset
from openvino._pyopenvino.op import Parameter
from openvino._pyopenvino.properties.hint import inference_precision

from nncf import CompressWeightsMode
from nncf.common.utils.backend import is_openvino_at_least
from nncf.common.utils.caching import ResultsCache
from nncf.common.utils.caching import cache_results
from nncf.common.utils.helpers import set_env_variable
from nncf.openvino.cpu_info import is_lnl_cpu
from nncf.openvino.graph.node_utils import convert_op
from nncf.openvino.graph.node_utils import non_convertable_divide_op
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.constants import FP_MAX_VALUES
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor.functions.openvino_numeric import DTYPE_MAP as DTYPE_MAP_OV

TensorList = list[Tensor]
ModelCallable = Callable[[TensorList], TensorList]
ReductionAxes = Union[int, tuple[int, ...]]


OV_MODEL_CACHE = ResultsCache()


@dataclass
class OVModelParameters:
    """
    A class to hold parameters for building and inferring an OpenVINO model.

    :param input_dtypes: Optional dictionary mapping input names to their data types.
    :param output_dtypes: Optional dictionary mapping output names to their data types.
    :param dynamic_shapes: Whether to use dynamic shapes for the model. When dynamic shapes are used and
        models are cached, it allows to save on the number of models stored in a model cache.
    :param release_memory: Whether to release memory after every inference. If memory is released, it will be
        reallocated during every inference, reducing performance to some extent.
    :param share_inputs: Whether to share input tensors. Avoids cloning inputs for inference.
    :param share_outputs: Whether to share output tensors. Avoids cloning outputs after the inference.
    :param return_ov_tensors: Whether to return results as OpenVINO tensors or NumPy arrays.
    :param convertable_division: Whether to use convertable division for division operations. If True, division a/b
        will be transformed at runtime to a*(1/b).
    """

    input_dtypes: dict[str, TensorDataType] = field(default_factory=dict)
    output_dtypes: dict[str, TensorDataType] = field(default_factory=dict)
    dynamic_shapes: bool = True
    release_memory: bool = True
    share_inputs: bool = True
    share_outputs: bool = True
    return_ov_tensors: bool = False
    convertable_division: bool = False

    def __copy__(self):
        return OVModelParameters(
            input_dtypes=self.input_dtypes.copy(),
            output_dtypes=self.output_dtypes.copy(),
            dynamic_shapes=self.dynamic_shapes,
            release_memory=self.release_memory,
            share_inputs=self.share_inputs,
            share_outputs=self.share_outputs,
            return_ov_tensors=self.return_ov_tensors,
            convertable_division=self.convertable_division,
        )

    def __deepcopy__(self, memo):
        return OVModelParameters(
            input_dtypes=copy.deepcopy(self.input_dtypes, memo),
            output_dtypes=copy.deepcopy(self.output_dtypes, memo),
            dynamic_shapes=self.dynamic_shapes,
            release_memory=self.release_memory,
            share_inputs=self.share_inputs,
            share_outputs=self.share_outputs,
            return_ov_tensors=self.return_ov_tensors,
            convertable_division=self.convertable_division,
        )

    def __hash__(self):
        return hash(
            (
                frozenset(self.input_dtypes.items()),
                frozenset(self.output_dtypes.items()),
                self.dynamic_shapes,
                self.release_memory,
                self.share_inputs,
                self.share_outputs,
                self.return_ov_tensors,
                self.convertable_division,
            )
        )

    def clone(self):
        return copy.deepcopy(self)


ModelAsNodes = tuple[list[Parameter], list[Node], OVModelParameters]


def clear_ov_model_cache():
    OV_MODEL_CACHE.clear()


def _compile_ov_model(model: ov.Model, device_name: str, config: dict[str, str]) -> ov.CompiledModel:
    if is_lnl_cpu() and not is_openvino_at_least("2025.1"):
        with set_env_variable("DNNL_MAX_CPU_ISA", "AVX2_VNNI"):
            compiled_model = ov.compile_model(model, device_name=device_name, config=config)
    else:
        compiled_model = ov.compile_model(model, device_name=device_name, config=config)

    return compiled_model


def _infer_ov_model(
    ov_model_params: OVModelParameters, compiled_model: ov.CompiledModel, inputs: TensorList
) -> TensorList:
    """
    Run compiled OpenVINO model inference on the given inputs.

    :param ov_model_params: OV model related parameters.
    :param compiled_model: Compiled OpenVINO model.
    :param inputs: Input tensors.
    :return: List of output tensors. Tensor backend is OV if return_ov_tensors is True, else NumPy.
    """
    # Check that input dtypes match the expected dtypes
    for i, inp in enumerate(compiled_model.inputs):
        input_name = inp.any_name
        actual_dtype = inputs[i].dtype
        expected_dtype = ov_model_params.input_dtypes[input_name]
        if actual_dtype != expected_dtype:
            msg = f"Expected input '{input_name}' to be {expected_dtype}. But found: {actual_dtype}."
            raise ValueError(msg)

    # Infer the model
    if compiled_model._infer_request is None:
        compiled_model._infer_request = compiled_model.create_infer_request()
    infer_request = compiled_model._infer_request

    inputs = [inp.data for inp in inputs]
    outputs = infer_request.infer(
        inputs, share_inputs=ov_model_params.share_inputs, share_outputs=ov_model_params.share_outputs
    )
    if ov_model_params.return_ov_tensors:
        outputs = [infer_request.get_output_tensor(i) for i in range(len(outputs))]
    else:
        outputs = [outputs[i] for i in range(len(outputs))]
    outputs = [Tensor(it) for it in outputs]

    if ov_model_params.release_memory:
        compiled_model.release_memory()

    return outputs


def _prepare_quantization_model_inputs(
    ov_model_params,
    weight_shape: tuple,
    scale_shape: Optional[tuple],
    zero_point_shape: Optional[tuple],
    reduction_axes: Optional[ReductionAxes],
) -> tuple[tuple, Optional[tuple], Optional[tuple]]:
    """
    Do some input checks and convert static shapes to dynamic shapes if needed.
    """
    if scale_shape is None and zero_point_shape is not None:
        msg = "Zero point shape can only be provided if scale shape is provided."
        raise Exception(msg)
    if scale_shape is None and reduction_axes is None:
        msg = "Reduction axes must be provided if scale shape is not provided."
        raise ValueError(msg)

    # Set dynamic shapes if needed
    if ov_model_params.dynamic_shapes:
        weight_shape = (-1,) * len(weight_shape)
        # TODO(andrey-churkin): This is a workaround to support an arbitrary reduction axis.
        # We should use `reduction_axes`` here and make it a non-optional parameter.
        if scale_shape is not None:
            scale_shape = tuple(-1 if d != 1 else d for d in scale_shape)
        if zero_point_shape is not None:
            zero_point_shape = tuple(-1 if d != 1 else d for d in zero_point_shape)

    return weight_shape, scale_shape, zero_point_shape


def _validate_input_dtypes(
    weight_dtype: TensorDataType,
    scale_shape: Optional[tuple] = None,
    input_scale_dtype: Optional[TensorDataType] = None,
    zero_point_shape: Optional[tuple] = None,
    input_zero_point_dtype: Optional[TensorDataType] = None,
):
    """
    Validates the input data types for the quantization model.

    Ensures that the weight data type is one of the supported floating-point types
    (float32, float16, bfloat16, f8e4m3, f8e5m2), and that the scale and zero point
    data types are float32 and int32/float32 respectively, if provided.

    :param weight_dtype: Data type of the weight tensor.
    :param scale_shape: Shape of the scale tensor (optional).
    :param input_scale_dtype: Data type of the scale tensor (optional).
    :param zero_point_shape: Shape of the zero point tensor (optional).
    :param input_zero_point_dtype: Data type of the zero point tensor (optional).
    :raises ValueError: If the weight data type is not supported, or if the scale
        or zero point data types are incorrect.
    """
    valid_weight_dtypes = [
        TensorDataType.float32,
        TensorDataType.float16,
        TensorDataType.bfloat16,
        TensorDataType.f8e4m3,
        TensorDataType.f8e5m2,
    ]
    if weight_dtype not in valid_weight_dtypes:
        msg = f"Weight must be one of the following data types: {valid_weight_dtypes}. But found: {weight_dtype}."
        raise ValueError(msg)
    if scale_shape is not None and input_scale_dtype != TensorDataType.float32:
        msg = f"Input scale must be of float32 data type. But found: {input_scale_dtype}."
        raise ValueError(msg)
    if zero_point_shape is not None and input_zero_point_dtype not in [TensorDataType.int32, TensorDataType.float32]:
        msg = f"Input zero point must be of int32/float32 data type. But found: {input_zero_point_dtype}."
        raise ValueError(msg)


def get_integer_quantization_model(
    ov_model_params: OVModelParameters,
    config: WeightCompressionConfig,
    weight_shape: tuple,
    scale_shape: Optional[tuple] = None,
    zero_point_shape: Optional[tuple] = None,
    reduction_axes: Optional[ReductionAxes] = None,
) -> Union[ModelCallable, ModelAsNodes]:
    """
    Get a model that compresses weights using the given configuration.

    :param ov_model_params: OV model parameters.
    :param config: Compression configuration.
    :param weight_shape: Shape of the weight to compress. Weight is assumed to be already reshaped as needed.
    :param scale_shape: Optional shape of the scale. If not provided, scale will be computed by the OV model.
        Otherwise, it is expected that the scale tensor is given as an input to the model.
    :param zero_point_shape: Optional shape of the zero point tensor. If not provided and the mode is asymmetric,
        zero point will be computed by the OV model. Otherwise, it is expected that the zero point tensor is provided
        as an input.
    :param reduction_axes: Optional axes to reduce the weight tensor. Not needed if scale (and z.p.) are provided as
        inputs.
    :return: A model callable that compresses weights using the given configuration. Or a model as nodes, if
        `return_nodes` is True.
    """
    weight_shape, scale_shape, zero_point_shape = _prepare_quantization_model_inputs(
        ov_model_params, weight_shape, scale_shape, zero_point_shape, reduction_axes
    )

    return _build_integer_quantization_model(
        config,
        ov_model_params,
        weight_shape,
        scale_shape,
        zero_point_shape,
        reduction_axes,
    )


def get_float_quantization_model(
    ov_model_params: OVModelParameters,
    config: WeightCompressionConfig,
    weight_shape: tuple,
    scale_shape: Optional[tuple] = None,
    reduction_axes: Optional[ReductionAxes] = None,
) -> Union[ModelCallable, ModelAsNodes]:
    """
    Get a model that compresses weights to float destination type using the given configuration.

    :param ov_model_params: OV model parameters.
    :param config: Compression configuration.
    :param weight_shape: Shape of the weight to compress. Weight is assumed to be already reshaped as needed.
    :param scale_shape: Optional shape of the scale. If not provided, scale will be computed by the OV model.
        Otherwise, it is expected that the scale tensor is given as an input to the model.
    :param reduction_axes: Optional axes to reduce the weight tensor. Not needed if scale is provided as input.
    :return: A model callable that compresses weights using the given configuration.
    """
    weight_shape, scale_shape, _ = _prepare_quantization_model_inputs(
        ov_model_params, weight_shape, scale_shape, zero_point_shape=None, reduction_axes=reduction_axes
    )

    return _build_float_quantization_model(
        config,
        ov_model_params,
        weight_shape,
        scale_shape,
        reduction_axes,
    )


def get_float_quantize_dequantize_weight_model(
    ov_model_params: OVModelParameters,
    config: WeightCompressionConfig,
    weight_shape: tuple,
    scale_shape: Optional[tuple] = None,
    reduction_axes: Optional[ReductionAxes] = None,
    return_compressed_weight: Optional[bool] = False,
) -> ModelCallable:
    """
    Get a model that performs float compression and decompression of the given weight.

    :param ov_model_params: OV model parameters.
    :param config: Compression configuration.
    :param weight_shape: Shape of the weight. Weight is assumed to be already reshaped as needed.
    :param scale_shape: Optional shape of the scale. If not provided, scale will be computed by the OV model.
        Otherwise, it is expected that the scale tensor is given as an input to the model.
    :param reduction_axes: Optional axes to reduce the weight tensor. Not needed if scale is provided as input.
    :param return_compressed_weight: Whether to also return compressed weight and scale besides the
        decompressed weight.
    :return: A model callable that returns a decompressed weight, and optionally compressed weight and scale
        if `return_compressed_weight` is True.
    """
    weight_shape, scale_shape, _ = _prepare_quantization_model_inputs(
        ov_model_params, weight_shape, scale_shape, zero_point_shape=None, reduction_axes=reduction_axes
    )

    return _build_float_quantize_dequantize_weight_model(
        config,
        ov_model_params,
        weight_shape,
        scale_shape,
        reduction_axes,
        return_compressed_weight,
    )


def get_integer_quantize_dequantize_weight_model(
    ov_model_params: OVModelParameters,
    config: WeightCompressionConfig,
    weight_shape: tuple,
    scale_shape: Optional[tuple] = None,
    zero_point_shape: Optional[tuple] = None,
    reduction_axes: Optional[ReductionAxes] = None,
    return_compressed_weight: Optional[bool] = False,
) -> ModelCallable:
    """
    Get a model that performs compression and decompression of the given weight.

    :param ov_model_params: OV model parameters.
    :param config: Compression configuration.
    :param weight_shape: Shape of the weight. Weight is assumed to be already reshaped as needed.
    :param scale_shape: Optional shape of the scale. If not provided, scale will be computed by the OV model.
        Otherwise, it is expected that the scale tensor is given as an input to the model.
    :param zero_point_shape: Optional shape of the zero point tensor. If not provided and the mode is asymmetric,
        zero point will be computed by the OV model. Otherwise, it is expected that the zero point is provided as an
        input.
    :param reduction_axes: Optional axes to reduce the weight tensor. Not needed if scale (and z.p.) are provided as
        inputs.
    :param return_compressed_weight: Whether to also return compressed weight, scale, (and zero point) besides the
        decompressed weight.
    :return: A model callable that returns a decompressed weight, and optionally compressed weight, scale,
        (and zero point) if `return_compressed_weight` is True.
    """
    weight_shape, scale_shape, zero_point_shape = _prepare_quantization_model_inputs(
        ov_model_params, weight_shape, scale_shape, zero_point_shape, reduction_axes
    )

    return _build_integer_quantize_dequantize_weight_model(
        config,
        ov_model_params,
        weight_shape,
        scale_shape,
        zero_point_shape,
        reduction_axes,
        return_compressed_weight,
    )


def get_integer_quantization_error_model(
    ov_model_params: OVModelParameters,
    config: WeightCompressionConfig,
    reduction: str,
    weight_shape: tuple,
    reduction_axes: ReductionAxes,
    original_weight_shape: tuple,
    original_reduction_axes: ReductionAxes,
) -> ModelCallable:
    """
    Get a model that calculates the quantization error for a given weight.

    This function builds a model that compresses and then decompresses the given weight, and calculates the
    quantization error by comparing the original weight with the decompressed weight.

    :param ov_model_params: OV model parameters.
    :param config: Compression configuration.
    :param reduction: Reduction mode to aggregate error values. Supported modes: "max_mean", "frobenius".
    :param weight_shape: Shape of the weight tensor to be compressed.
    :param reduction_axes: Axes to reduce the weight tensor.
    :param original_weight_shape: Shape of the original weight tensor.
    :param original_reduction_axes: Reduction axes of the original weight tensor before reshaping.
    :return: A model callable that returns the quantization error.
    """
    weight_shape, _, _ = _prepare_quantization_model_inputs(ov_model_params, weight_shape, None, None, reduction_axes)

    return _build_integer_quantization_error_model(
        config,
        ov_model_params,
        reduction,
        weight_shape,
        reduction_axes,
        original_weight_shape,
        original_reduction_axes,
    )


@cache_results(OV_MODEL_CACHE)
def _build_integer_quantization_model(
    config: WeightCompressionConfig,
    ov_model_params: OVModelParameters,
    weight_shape: tuple,
    scale_shape: Optional[tuple] = None,
    zero_point_shape: Optional[tuple] = None,
    reduction_axes: Optional[ReductionAxes] = None,
    return_nodes: bool = False,
) -> Union[ModelCallable, ModelAsNodes]:
    is_asym_mode = config.is_asym_mode

    default_input_dtypes = {
        "scale": TensorDataType.float32,
        "zero_point": TensorDataType.int32,
    }
    default_output_dtypes = {
        "compressed_weight": TensorDataType.uint8 if is_asym_mode else TensorDataType.int8,
        "scale": TensorDataType.float32,
        "zero_point": TensorDataType.int32,
    }

    # Update input and output dtypes with the default values
    ov_model_params = copy.deepcopy(ov_model_params)
    ov_model_params.input_dtypes = {**default_input_dtypes, **ov_model_params.input_dtypes}
    ov_model_params.output_dtypes = {**default_output_dtypes, **ov_model_params.output_dtypes}

    if "weight" not in ov_model_params.input_dtypes:
        msg = "Input weight dtype is required!"
        raise ValueError(msg)

    weight_dtype = ov_model_params.input_dtypes["weight"]
    input_scale_dtype = ov_model_params.input_dtypes["scale"]
    input_zero_point_dtype = ov_model_params.input_dtypes["zero_point"]
    compressed_weight_dtype = ov_model_params.output_dtypes["compressed_weight"]
    output_scale_dtype = ov_model_params.output_dtypes["scale"]
    output_zero_point_dtype = ov_model_params.output_dtypes["zero_point"]

    _validate_input_dtypes(
        weight_dtype=weight_dtype,
        scale_shape=scale_shape,
        input_scale_dtype=input_scale_dtype,
        zero_point_shape=zero_point_shape,
        input_zero_point_dtype=input_zero_point_dtype,
    )

    # Validate output dtypes
    valid_compressed_weight_dtypes = [
        TensorDataType.float32,
        TensorDataType.int32,
        TensorDataType.int8,
        TensorDataType.uint8,
        TensorDataType.int4,
        TensorDataType.uint4,
    ]
    if compressed_weight_dtype not in valid_compressed_weight_dtypes:
        msg = (
            f"Compressed weight must be one of the following data types: {valid_compressed_weight_dtypes}. "
            f"But found: {compressed_weight_dtype}."
        )
        raise ValueError(msg)
    if scale_shape is None and output_scale_dtype != TensorDataType.float32:
        msg = f"Output scale must be of float32 data type. But found: {output_scale_dtype}."
        raise ValueError(msg)
    if is_asym_mode and zero_point_shape is None and output_zero_point_dtype not in valid_compressed_weight_dtypes:
        msg = (
            f"Output zero point must be of one of the following data types: {valid_compressed_weight_dtypes}. "
            f"But found: {output_zero_point_dtype}."
        )
        raise ValueError(msg)

    # Build OV model
    weight = opset.parameter(weight_shape, name="weight", dtype=DTYPE_MAP_OV[weight_dtype])
    ov_parameters = [weight]

    num_bits = config.num_bits
    eps = np.finfo(np.float32).eps
    level_low = 0 if is_asym_mode else -(2 ** (num_bits - 1))
    level_high = 2**num_bits - 1 if is_asym_mode else 2 ** (num_bits - 1) - 1

    divide_op = opset.divide if ov_model_params.convertable_division else non_convertable_divide_op

    min_values = None
    if scale_shape is not None:
        # Scale is given as an input
        scale = opset.parameter(scale_shape, name="scale", dtype=DTYPE_MAP_OV[input_scale_dtype])
        ov_parameters.append(scale)
    else:
        # Compute scale
        # [a1, r, a2] -> [a1, 1, a2]
        min_values = opset.reduce_min(weight, reduction_axes=reduction_axes, keep_dims=True)
        max_values = opset.reduce_max(weight, reduction_axes=reduction_axes, keep_dims=True)
        min_values, max_values = opset.convert(min_values, ov.Type.f32), opset.convert(max_values, ov.Type.f32)

        if is_asym_mode:
            levels = level_high - level_low + 1
            scale = divide_op(max_values - min_values, opset.constant(levels - 1, ov.Type.f32))
            scale = opset.select(opset.less(opset.abs(scale), eps), eps, scale)
        else:
            abs_min_values = opset.abs(min_values)

            scale = opset.select(
                opset.greater_equal(abs_min_values, max_values), abs_min_values, opset.negative(max_values)
            )
            scale = divide_op(scale, opset.constant(-level_low, ov.Type.f32))
            scale = opset.select(opset.less(opset.abs(scale), eps), eps, scale)

    zero_point = None
    if zero_point_shape is not None:
        # Zero point is given as an input
        zero_point = opset.parameter(zero_point_shape, name="zero_point", dtype=DTYPE_MAP_OV[input_zero_point_dtype])
        ov_parameters.append(zero_point)
        # Cast to float32 for an addition later
        zero_point = convert_op(zero_point, ov.Type.f32)
    elif is_asym_mode:
        # Compute zero point
        scaled_min_values = divide_op(min_values, scale)
        zero_point = opset.constant(level_low, ov.Type.f32) - opset.round(scaled_min_values)
        zero_point = opset.clamp(zero_point, level_low, level_high)

    weight = convert_op(weight, ov.Type.f32)
    compressed_weight = divide_op(weight, scale)

    if is_asym_mode:
        compressed_weight += zero_point

    compressed_weight = opset.round(compressed_weight)
    compressed_weight = opset.clamp(opset.round(compressed_weight), level_low, level_high)
    compressed_weight = convert_op(compressed_weight, DTYPE_MAP_OV[compressed_weight_dtype])

    ov_results = [compressed_weight]
    if len(ov_parameters) == 1:
        ov_results.append(scale)
        if zero_point is not None:
            zero_point = convert_op(zero_point, DTYPE_MAP_OV[output_zero_point_dtype])
            ov_results.append(zero_point)

    if return_nodes:
        return ov_parameters, ov_results, ov_model_params

    model = ov.Model(ov_results, ov_parameters)
    compiled_model = _compile_ov_model(model, device_name="CPU", config={inference_precision(): ov.Type.f32})

    return partial(_infer_ov_model, ov_model_params, compiled_model)


@cache_results(OV_MODEL_CACHE)
def _build_float_quantization_model(
    config: WeightCompressionConfig,
    ov_model_params: OVModelParameters,
    weight_shape: tuple,
    scale_shape: Optional[tuple] = None,
    reduction_axes: Optional[ReductionAxes] = None,
    return_nodes: bool = False,
) -> Union[ModelCallable, ModelAsNodes]:
    default_input_dtypes = {"scale": TensorDataType.float32}
    default_output_dtypes = {"compressed_weight": TensorDataType.float32, "scale": TensorDataType.float32}

    # Update input and output dtypes with the default values
    ov_model_params = copy.deepcopy(ov_model_params)
    ov_model_params.input_dtypes = {**default_input_dtypes, **ov_model_params.input_dtypes}
    ov_model_params.output_dtypes = {**default_output_dtypes, **ov_model_params.output_dtypes}

    if "weight" not in ov_model_params.input_dtypes:
        msg = "Input weight dtype is required!"
        raise ValueError(msg)

    weight_dtype = ov_model_params.input_dtypes["weight"]
    input_scale_dtype = ov_model_params.input_dtypes["scale"]
    compressed_weight_dtype = ov_model_params.output_dtypes["compressed_weight"]
    output_scale_dtype = ov_model_params.output_dtypes["scale"]

    _validate_input_dtypes(
        weight_dtype=weight_dtype,
        scale_shape=scale_shape,
        input_scale_dtype=input_scale_dtype,
    )

    # Validate output dtypes
    valid_compressed_weight_dtypes = [
        TensorDataType.float32,
        TensorDataType.nf4,
        TensorDataType.f4e2m1,
        TensorDataType.f8e4m3,
    ]
    if compressed_weight_dtype not in valid_compressed_weight_dtypes:
        msg = (
            f"Compressed weight must be one of the following data types: {valid_compressed_weight_dtypes}. "
            f"But found: {compressed_weight_dtype}."
        )
        raise ValueError(msg)
    if scale_shape is None and output_scale_dtype != TensorDataType.float32:
        msg = f"Output scale must be of float32 data type. But found: {output_scale_dtype}."
        raise ValueError(msg)

    # Build OV model
    weight = opset.parameter(weight_shape, name="weight", dtype=DTYPE_MAP_OV[weight_dtype])
    ov_parameters = [weight]
    weight = convert_op(weight, ov.Type.f32)

    divide_op = opset.divide if ov_model_params.convertable_division else non_convertable_divide_op
    if scale_shape is not None:
        # Scale is given as an input
        scale = opset.parameter(scale_shape, name="scale", dtype=DTYPE_MAP_OV[input_scale_dtype])
        ov_parameters.append(scale)
    else:
        # Compute scale
        scale = opset.reduce_max(opset.abs(weight), reduction_axes=reduction_axes, keep_dims=True)
        # NOTE: adding machine epsilon to avoid division by zero
        eps = np.finfo(np.float32).eps
        scale = opset.select(opset.less(opset.abs(scale), eps), eps, scale)

        if config.compression_dtype != TensorDataType.nf4:
            scale = divide_op(scale, opset.constant(FP_MAX_VALUES[config.compression_dtype], ov.Type.f32))

        if config.mode in [CompressWeightsMode.MXFP4, CompressWeightsMode.MXFP8_E4M3]:
            scale = opset.log(scale) / opset.log(opset.constant(2.0, ov.Type.f32))
            scale = opset.ceil(scale)
            scale = opset.clamp(scale, -127.0, 127.0)
            scale = opset.power(opset.constant(2.0, ov.Type.f32), scale)

    compressed_weight = divide_op(weight, scale)
    compressed_weight = convert_op(compressed_weight, DTYPE_MAP_OV[config.compression_dtype])
    compressed_weight = convert_op(compressed_weight, DTYPE_MAP_OV[compressed_weight_dtype])

    ov_results = [compressed_weight]
    if len(ov_parameters) == 1:
        ov_results.append(scale)

    if return_nodes:
        return ov_parameters, ov_results, ov_model_params

    model = ov.Model(ov_results, ov_parameters)
    compiled_model = _compile_ov_model(model, device_name="CPU", config={inference_precision(): ov.Type.f32})

    return partial(_infer_ov_model, ov_model_params, compiled_model)


@cache_results(OV_MODEL_CACHE)
def _build_integer_quantize_dequantize_weight_model(
    config: WeightCompressionConfig,
    ov_model_params: OVModelParameters,
    weight_shape: tuple,
    scale_shape: Optional[tuple] = None,
    zero_point_shape: Optional[tuple] = None,
    reduction_axes: Optional[ReductionAxes] = None,
    return_compressed_weight: Optional[bool] = False,
    return_nodes: Optional[bool] = False,
) -> Union[ModelCallable, ModelAsNodes]:
    default_output_dtypes = {"decompressed_weight": TensorDataType.float32}
    if not return_compressed_weight:
        # If compressed weight is not returned to a user, we can keep it in float32 to avoid additional conversion
        default_output_dtypes["compressed_weight"] = TensorDataType.float32
    ov_model_params = copy.deepcopy(ov_model_params)
    ov_model_params.output_dtypes = {**default_output_dtypes, **ov_model_params.output_dtypes}

    decompressed_weight_dtype = ov_model_params.output_dtypes["decompressed_weight"]
    if decompressed_weight_dtype != TensorDataType.float32:
        msg = f"Decompressed weight must be of float32 data type. But found: {decompressed_weight_dtype}."
        raise ValueError(msg)

    # Get compression model as input/result nodes and potentially modified ov model parameters
    ov_parameters, ov_results, ov_model_params = _build_integer_quantization_model(
        config, ov_model_params, weight_shape, scale_shape, zero_point_shape, reduction_axes, return_nodes=True
    )

    if config.is_asym_mode:
        if len(ov_parameters) == 1:
            # weight -> compressed_weight, scale, zero_point
            compressed_weight, scale, zero_point = ov_results
        else:
            # weight, scale, zero_point -> compressed_weight
            compressed_weight = ov_results[0]
            scale, zero_point = ov_parameters[1:]

        compressed_weight = convert_op(compressed_weight, ov.Type.i32) - convert_op(zero_point, ov.Type.i32)
    else:
        if len(ov_parameters) == 1:
            # weight -> compressed_weight, scale
            compressed_weight, scale = ov_results
        else:
            # weight, scale -> compressed_weight
            compressed_weight = ov_results[0]
            scale = ov_parameters[1]

    decompressed_weight = opset.multiply(convert_op(compressed_weight, ov.Type.f32), scale)

    ov_results = [decompressed_weight] + ov_results if return_compressed_weight else [decompressed_weight]

    if return_nodes:
        return ov_parameters, ov_results, ov_model_params

    model = ov.Model(ov_results, ov_parameters)
    compiled_model = _compile_ov_model(model, device_name="CPU", config={inference_precision(): ov.Type.f32})

    return partial(_infer_ov_model, ov_model_params, compiled_model)


@cache_results(OV_MODEL_CACHE)
def _build_float_quantize_dequantize_weight_model(
    config: WeightCompressionConfig,
    ov_model_params: OVModelParameters,
    weight_shape: tuple,
    scale_shape: Optional[tuple] = None,
    reduction_axes: Optional[ReductionAxes] = None,
    return_compressed_weight: Optional[bool] = False,
    return_nodes: Optional[bool] = False,
) -> Union[ModelCallable, ModelAsNodes]:
    default_output_dtypes = {"decompressed_weight": TensorDataType.float32}
    if not return_compressed_weight:
        # If compressed weight is not returned to a user, we can keep it in float32 to avoid additional conversion
        default_output_dtypes["compressed_weight"] = TensorDataType.float32
    ov_model_params = copy.deepcopy(ov_model_params)
    ov_model_params.output_dtypes = {**default_output_dtypes, **ov_model_params.output_dtypes}

    decompressed_weight_dtype = ov_model_params.output_dtypes["decompressed_weight"]
    if decompressed_weight_dtype != TensorDataType.float32:
        msg = f"Decompressed weight must be of float32 data type. But found: {decompressed_weight_dtype}."
        raise ValueError(msg)

    # Get compression model as input/result nodes and potentially modified ov model parameters
    ov_parameters, ov_results, ov_model_params = _build_float_quantization_model(
        config, ov_model_params, weight_shape, scale_shape, reduction_axes, return_nodes=True
    )

    if len(ov_parameters) == 1:
        # weight -> compressed_weight, scale
        compressed_weight, scale = ov_results
    else:
        # weight, scale -> compressed_weight
        compressed_weight = ov_results[0]
        scale = ov_parameters[1]

    decompressed_weight = opset.multiply(scale, convert_op(compressed_weight, ov.Type.f32))

    ov_results = [decompressed_weight] + ov_results if return_compressed_weight else [decompressed_weight]

    if return_nodes:
        return ov_parameters, ov_results, ov_model_params

    model = ov.Model(ov_results, ov_parameters)
    compiled_model = _compile_ov_model(model, device_name="CPU", config={inference_precision(): ov.Type.f32})

    return partial(_infer_ov_model, ov_model_params, compiled_model)


@cache_results(OV_MODEL_CACHE)
def _build_integer_quantization_error_model(
    config: WeightCompressionConfig,
    ov_model_params: OVModelParameters,
    reduction: str,
    weight_shape: tuple,
    reduction_axes: ReductionAxes,
    original_weight_shape: tuple,
    original_reduction_axes: ReductionAxes,
) -> ModelCallable:
    ov_parameters, ov_results, ov_model_params = _build_integer_quantize_dequantize_weight_model(
        config,
        ov_model_params,
        weight_shape,
        reduction_axes=reduction_axes,
        return_compressed_weight=False,
        return_nodes=True,
    )

    weight = ov_parameters[0]
    decompressed_weight = ov_results[0]

    weight = convert_op(weight, ov.Type.f32)
    if reduction == "max_mean":
        weight = opset.reshape(weight, original_weight_shape, special_zero=False)
        decompressed_weight = opset.reshape(decompressed_weight, original_weight_shape, special_zero=False)
        diff = opset.squared_difference(decompressed_weight, weight)
        layer_err = opset.reduce_mean(diff, reduction_axes=original_reduction_axes)
        quantization_error = opset.reduce_max(layer_err, reduction_axes=tuple(range(len(layer_err.shape))))
    elif reduction == "frobenius":
        diff = opset.reshape(decompressed_weight - weight, (-1,), special_zero=False)
        quantization_error = opset.matmul(diff, diff, transpose_a=False, transpose_b=False)
        quantization_error = opset.sqrt(quantization_error)
    else:
        msg = f"Unsupported aggregation method: {reduction}."
        raise ValueError(msg)

    model = ov.Model([quantization_error], ov_parameters)
    compiled_model = _compile_ov_model(model, device_name="CPU", config={inference_precision(): ov.Type.f32})

    return partial(_infer_ov_model, ov_model_params, compiled_model)


def get_astype_model(ov_model_params: OVModelParameters, input_shape: tuple) -> ModelCallable:
    """
    Return a model that cast the input of the given shape to the given data type. Especially useful for
    casting from/to data types not supported by NumPy such as bfloat16, uint4 and int4.
    These data types are represented as the following data types in numpy:
        - bfloat16 -> np.float16,
        - uint4 -> uint8,
        - int4 -> int8.

    :param ov_model_params: OV model related parameters.
    :param input_shape: Shape of the tensor to cast.
    :return: A model callable that casts the input tensor to the given data type.
    """
    if ov_model_params.dynamic_shapes:
        input_shape = (-1,) * len(input_shape)
    return _build_astype_model(ov_model_params, input_shape)


@cache_results(OV_MODEL_CACHE)
def _build_astype_model(ov_model_params: OVModelParameters, arg_shape: tuple) -> ModelCallable:
    input_dtypes = ov_model_params.input_dtypes
    if input_dtypes is None:
        msg = "Input dtypes must be provided."
        raise ValueError(msg)
    output_dtypes = ov_model_params.output_dtypes
    if output_dtypes is None:
        msg = "Output dtypes must be provided."
        raise ValueError(msg)
    if "input" not in input_dtypes:
        msg = "Input dtype is required."
        raise ValueError(msg)
    if "output" not in output_dtypes:
        msg = "Output dtype is required."
        raise ValueError(msg)

    arg = opset.parameter(arg_shape, dtype=DTYPE_MAP_OV[input_dtypes["input"]], name="input")
    res = opset.convert(arg, DTYPE_MAP_OV[output_dtypes["output"]])
    model = ov.Model([res], [arg])
    compiled_model = _compile_ov_model(model, device_name="CPU", config={inference_precision(): ov.Type.f32})

    return partial(_infer_ov_model, ov_model_params, compiled_model)
