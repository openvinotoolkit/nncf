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

import numpy as np
import openvino.runtime as ov
from nncf.openvino.graph.metatypes.openvino_metatypes import OV_OPERATOR_METATYPES
from nncf.openvino.graph.metatypes.openvino_metatypes import get_operation_const_op
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVEmbeddingMetatype
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.statistics.statistics import OVMinMaxTensorStatistic
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.quantization.fake_quantize import calculate_quantizer_parameters
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import  QuantizerGroup
from openvino.runtime import opset9 as opset


def insert_pre_compression_operations(model: ov.Model, bits: int = 8):
    """
    Inserts in-place weights compression with FakeQuantize operation for Linear and Embedding layers.

    :param model: The original model to insert the weights compression.
    :param bits: number of bits for compression/quantization. Note: compressed weights type is
        uint8 with one element per 8 bit.
    """
    allowed_metatypes_to_const_port = {OVEmbeddingMetatype: [0], OVMatMulMetatype: [0, 1]}

    for node in model.get_ops():
        node_type = node.get_type_name()
        metatype = OV_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node_type)
        if metatype not in allowed_metatypes_to_const_port:
             continue

        stats = None
        for const_port_id in allowed_metatypes_to_const_port[metatype]:
            constant_node = get_operation_const_op(node, const_port_id)
            if constant_node is None:
                continue

            if metatype is OVMatMulMetatype:
                attribute_names = ["transpose_a", "transpose_b"]
                node_attributes = node.get_attributes()
                const_transpose_name = attribute_names[const_port_id]
                transpose = node_attributes[const_transpose_name]
                target_dim = -2 if (const_port_id == 1) == transpose else -1

                ndims = len(node.input(const_port_id).get_partial_shape().get_max_shape())
                channel_axes = list(range(ndims - 2)) if ndims > 2 else []
                target_dim = max(ndims, 2) + target_dim
                if target_dim < ndims:
                    channel_axes.append(target_dim)
                axes = tuple(i for i in range(ndims) if i not in channel_axes)
            else:
                axes = 1

            weight = get_const_value(constant_node)
            input_low = np.min(weight, axis=axes, keepdims=True)
            input_high = np.max(weight, axis=axes, keepdims=True)
            stats = OVMinMaxTensorStatistic(input_low, input_high)

        if stats is None:
            continue

        quantizer_config = QuantizerConfig(
            num_bits=bits,
            mode=QuantizationMode.ASYMMETRIC,
            signedness_to_force=None,
            per_channel=True,
        )
        fq_params = calculate_quantizer_parameters(
            stats,
            quantizer_config,
            QuantizerGroup.WEIGHTS,
            narrow_range=False,
            half_range=False
        )

        inp_node = node.input(const_port_id)
        data_type = inp_node.get_element_type()
        if data_type == ov.Type(np.float16):
            input_low, input_high, output_low, output_high = OVModelTransformer.convert_params_to_fp16(fq_params)
        else:
            input_low = fq_params.input_low
            input_high = fq_params.input_high
            output_low = fq_params.output_low
            output_high = fq_params.output_high
        levels = fq_params.levels
        node_name = node.get_friendly_name()
        fq_name = f"{node_name}/fq_weights_{const_port_id}"
        fq = None
        input_node_output = inp_node.get_source_output()
        for out in input_node_output.get_target_inputs():
            if out.get_node().get_type_name() == "FakeQuantize":
                fq = out.get_node()
        if fq is None:
            fq = opset.fake_quantize(
                input_node_output, input_low, input_high, output_low, output_high, levels, name=fq_name
            )
        inp_node.replace_source_output(fq.output(0))
