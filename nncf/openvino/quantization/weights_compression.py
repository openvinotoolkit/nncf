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

from typing import Tuple, Type, Union

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset9 as opset

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVEmbeddingMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import get_node_metatype
from nncf.openvino.graph.metatypes.openvino_metatypes import get_operation_const_op
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.graph.node_utils import get_matmul_channel_axes
from nncf.quantization.fake_quantize import calculate_scale_zero_point


def insert_pre_compression_operations(model: ov.Model, bits: int = 8) -> None:
    """
    Compress weights of Linear and Embedding layers to uint8.
    The result of compression is the same as asymmetric weight quantization.

    :param model: The model to be transformed.
    :param bits: Number of bits for quantization.
    """
    allowed_metatypes_to_const_port = {OVEmbeddingMetatype: [0], OVMatMulMetatype: [0, 1]}
    level_low = 0
    level_high = 2**bits - 1

    for node in model.get_ops():
        metatype = get_node_metatype(node)
        if metatype not in allowed_metatypes_to_const_port:
            continue

        for const_port_id in allowed_metatypes_to_const_port[metatype]:
            weight_node = get_operation_const_op(node, const_port_id)
            if weight_node is None:
                continue

            weight_output = weight_node.output(0)
            weight_name = weight_node.get_friendly_name()
            target_inputs = weight_output.get_target_inputs()

            original_weight_dtype = weight_output.get_element_type().to_dtype()
            if original_weight_dtype not in [np.float32, np.float16, np.float64]:
                continue

            weight = get_const_value(weight_node)
            axes = _get_reduction_axes(metatype, node, const_port_id)
            min_values = np.min(weight, axis=axes, keepdims=True)
            max_values = np.max(weight, axis=axes, keepdims=True)

            scale, zero_point = calculate_scale_zero_point(
                min_values, max_values, level_low, level_high, narrow_range=False
            )

            compressed_weights = np.round(weight / scale + zero_point)
            compressed_weights = np.clip(compressed_weights, level_low, level_high).astype(np.uint8)

            compressed_const = opset.constant(compressed_weights, dtype=np.uint8, name=weight_name)
            convert = opset.convert(compressed_const, original_weight_dtype)
            sub = opset.subtract(convert, zero_point.astype(original_weight_dtype))
            fq_name = f"{node.get_friendly_name()}/fq_weights_{const_port_id}"
            mul = opset.multiply(sub, scale.astype(original_weight_dtype), name=fq_name)

            for target_input in target_inputs:
                target_input.replace_source_output(mul.output(0))


def _get_reduction_axes(metatype: Type[OperatorMetatype], node: ov.Node, weight_port_id: int) -> Union[int, Tuple[int]]:
    """
    Determines reduction axes by given metatype and node information.

    :param metatype: The metatype of the operator.
    :param node: The OpenVINO node.
    :param weight_port_id: The weight port ID.

    :return: The reduction axes as an integer or a tuple of integers.
    """
    if metatype is OVMatMulMetatype:
        transpose = node.get_attributes()[f"transpose_{'a' if weight_port_id == 0 else 'b'}"]
        ndims = node.input(weight_port_id).get_partial_shape().rank.get_max_length()
        channel_axes = get_matmul_channel_axes(weight_port_id, ndims, transpose)
        axes = tuple(i for i in range(ndims) if i not in channel_axes)
    elif metatype is OVEmbeddingMetatype:
        axes = (metatype.const_channel_axis[0] + 1) % 2
    else:
        RuntimeError("Unsupported metatype to find reduction axes.")
    return axes
