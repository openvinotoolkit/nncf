# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict

import openvino.runtime as ov
from openvino.runtime import opset13 as opset

import nncf
from nncf.common.graph import NNCFNode
from nncf.openvino.graph.metatypes import openvino_metatypes as om
from nncf.openvino.graph.model_utils import update_tensor_name
from nncf.openvino.graph.node_utils import get_parameter_node_name
from nncf.openvino.graph.node_utils import get_result_node_name


def build_for_fast_bc(
    model: ov.Model,
    node: NNCFNode,
    act_port_id: int,
    weight_port_id: int,
    out_port_id: int = 0,
    node_mapping=Dict[str, ov.Node],
) -> ov.Model:
    """
    Builds submodel for the FastBiasCorrection algorithm.
    The submodel consists of the biased layer (but without bias), weight quantized and weights:
                 Constant
                    |
    Parameter  FakeQuantize
        \          /
        Convolution
            |
          Result

    :param model: ov.Model instance as the reference.
    :param node: NNCFNode with the layer-related information.
    :param act_port_id: Activation port ID.
    :param weight_port_id: Weight port ID.
    :param out_port_id: Output port ID.
    :return: ov.Model subgraph.
    """
    # Create nodes mapping
    node_name = node.node_name
    original_node = node_mapping[node_name]
    activation_port = original_node.input_value(act_port_id)
    weight_port = original_node.input_value(weight_port_id)
    original_weight_fq = weight_port.get_node()
    weight_fq_in, weight_fq_in_low, weight_fq_in_high, weight_fq_out_low, weight_fq_out_high = [
        p.get_node() for p in original_weight_fq.input_values()
    ]
    # Build subgraph
    parameter_name = get_parameter_node_name(node_name, act_port_id)
    parameter = opset.parameter(
        shape=activation_port.partial_shape,
        dtype=activation_port.get_element_type(),
        name=parameter_name,
    )
    weight_fq_params = original_weight_fq.get_attributes()
    weight_fq_params.update(
        {
            "data": weight_fq_in,
            "input_low": weight_fq_in_low,
            "input_high": weight_fq_in_high,
            "output_low": weight_fq_out_low,
            "output_high": weight_fq_out_high,
            "name": original_weight_fq.get_friendly_name(),
        }
    )
    weights_fq = opset.fake_quantize(**weight_fq_params)
    main_node_params = original_node.get_attributes()
    if node.metatype == om.OVConvolutionMetatype:
        main_node_params.update({"data": parameter, "filters": weights_fq, "name": original_node.get_friendly_name()})
        main_node = opset.convolution(**main_node_params)
    elif node.metatype == om.OVMatMulMetatype:
        main_node_params.update({"data_a": parameter, "data_b": weights_fq, "name": original_node.get_friendly_name()})
        main_node = opset.matmul(**main_node_params)
    else:
        raise nncf.ModuleNotFoundError(f"Not found node type: {node.metatype.name}!")
    result_name = get_result_node_name(node_name, port_id=out_port_id)
    result = opset.result(main_node, name=result_name)
    update_tensor_name([result.get_output_tensor(0)], result_name)
    return ov.Model([result], [parameter])
