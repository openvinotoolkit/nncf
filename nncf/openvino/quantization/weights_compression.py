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
from functools import partial
import openvino.runtime as ov
from openvino.runtime import opset9 as opset
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import get_operation_const_op
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVEmbeddingMetatype
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.statistics.statistics import OVMinMaxTensorStatistic
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.quantization.fake_quantize import calculate_quantizer_parameters
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import  QuantizerGroup


def insert_pre_compression_operations(model: ov.Model, bits: int = 8):
    """
    Inserts in-place weights compression with FakeQuantize operation for Linear and Embedding layers.

    :param model: The original model to insert the weights compression.
    :param bits: Number of bits for quantization.
    """
    allowed_metatypes_to_const_port = {OVEmbeddingMetatype: [0], OVMatMulMetatype: [0, 1]}
    quantizer_config = QuantizerConfig(
        num_bits=bits,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=None,
        per_channel=True,
    )

    get_fq_params = partial(
        calculate_quantizer_parameters,
        quantizer_config=quantizer_config,
        quant_group=QuantizerGroup.WEIGHTS,
        narrow_range=False,
        half_range=False
    )

    for node in model.get_ops():
        metatype = GraphConverter._get_node_metatype(node)
        if metatype not in allowed_metatypes_to_const_port:
            continue

        for const_port_id in allowed_metatypes_to_const_port[metatype]:
            weight_node = get_operation_const_op(node, const_port_id)
            if weight_node is None:
                continue

            weight_output = weight_node.output(0)
            fq_count = 0
            for target_input in weight_output.get_target_inputs():
                consumer_node = target_input.get_node()
                if consumer_node.get_type_name() == "FakeQuantize":
                    fq_count += 1

            if fq_count > 0:
                # FQ must be linked with all target inputs
                assert fq_count == len(weight_output.get_target_inputs())
                continue

            weight = get_const_value(weight_node)
            axes = _get_reduction_axes(metatype, node, const_port_id)
            input_low = np.min(weight, axis=axes, keepdims=True)
            input_high = np.max(weight, axis=axes, keepdims=True)
            stats = OVMinMaxTensorStatistic(input_low, input_high)
            fq_params = get_fq_params(stats)

            node_name = node.get_friendly_name()
            fq_name = f"{node_name}/fq_weights_{const_port_id}"
            _insert_fake_quantize(fq_params, weight_output, fq_name)


def _get_reduction_axes(metatype: Type[OperatorMetatype], node: ov.Node, const_port_id: int) -> Union[int, Tuple[int]]:
    """
    Determines reduction axes by given metatype and node information.

    :param metatype: The metatype of the operator.
    :param node: The OpenVINO node.
    :param const_port_id : The weight port ID.

    :return: The reduction axes as an integer or a tuple of integers.
    """
    if metatype is OVMatMulMetatype:
        transpose = node.get_attributes()[f"transpose_{'a' if const_port_id == 0 else 'b'}"]
        ndims = node.input(const_port_id).get_partial_shape().rank.get_max_length()
        target_dim = -2 if (const_port_id == 1) and transpose else -1
        target_dim = max(ndims, 2) + target_dim
        channel_axes = list(range(ndims - 2))
        if target_dim < ndims:
            channel_axes.append(target_dim)
        axes = tuple(i for i in range(ndims) if i not in channel_axes)
    else:
        axes = 1
    return axes


def _insert_fake_quantize(fq_params: FakeQuantizeParameters, weight_output: ov.Output, fq_name: str):
    """
    Inserts a FakeQuantize operation into the model based on the given parameters.

    :param fq_params: FakeQuantize parameters.
    :param weight_output: Output of OpenVINO node.
    :param fq_name : Name for the inserted FakeQuantize operation.
    """
    if weight_output.get_element_type() == ov.Type(np.float16):
        input_low, input_high, output_low, output_high = OVModelTransformer.convert_params_to_fp16(fq_params)
    else:
        input_low = fq_params.input_low
        input_high = fq_params.input_high
        output_low = fq_params.output_low
        output_high = fq_params.output_high
    levels = fq_params.levels

    target_inputs = weight_output.get_target_inputs()
    fq = opset.fake_quantize(
        weight_output, input_low, input_high, output_low, output_high, levels, name=fq_name
    )

    for target_input in target_inputs:
        target_input.replace_source_output(fq.output(0))
