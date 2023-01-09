"""
 Copyright (c) 2022 Intel Corporation
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

import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype

from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OV_OPERATION_METATYPES
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVParameterMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVResultMetatype


class GraphConverter:
    """
    Builds the NNCFGraph from an OpenVINO model.
    """

    @staticmethod
    def convert_ov_dtype_to_nncf_dtype(ov_dtype: str) -> Dtype:
        """
        Converts the primitive types from the OpenVINO domain to the NNCF domain.

        :param ov_dtype: OpenVINO primitive typename.
        :return: NNCF primitive type.
        """
        conversion_map = {
            'f16': 'float',
            'f32': 'float',
            'i4': 'int',
            'i8': 'int',
            'i32': 'int',
            'i64': 'int',
            'u1': 'int',
            'u4': 'int',
            'u8': 'int',
            'u32': 'int',
            'u64': 'int',
        }
        if ov_dtype not in conversion_map:
            raise NotImplementedError(f'NNCF is not yet supported OpenVINO data type: {ov_dtype}.')
        return Dtype(conversion_map[ov_dtype])

    @staticmethod
    def create_nncf_graph(model: ov.Model) -> NNCFGraph:
        """
        Creates NNCFGraph from OpenVINO Model.
        All nodes from model which have valid metatype are added to NNCFGraph.
        Then, corresponding edges are added to the NNCFGraph with shape, type, output and input port ids.

        :param model: OpenVINO model.
        :return: NNCFGraph.
        """
        nncf_graph = NNCFGraph()

        op_name_to_node_id_map = {}
        for op in model.get_ops():
            op_name = op.get_friendly_name()
            op_type_name = op.get_type_name()
            metatype = OV_OPERATION_METATYPES.get_operator_metatype_by_op_name(op_type_name)

            nncf_node = nncf_graph.add_nncf_node(
                node_name=op_name,
                node_type=op_type_name,
                node_metatype=metatype
            )

            op_name_to_node_id_map[op_name] = nncf_node.node_id

        for producer_op in model.get_ops():
            for output in producer_op.outputs():
                output_port_id = output.get_index()
                tensor_shape = list(output.shape)
                tensor_dtype = output.get_element_type().get_type_name()

                # List of the (consumer_op, input_port_id) pairs
                consumer_ops = [
                    (input.get_node(), input.get_index()) for input in output.get_target_inputs()
                ]

                for consumer_op, input_port_id in consumer_ops:
                    nncf_graph.add_edge_between_nncf_nodes(
                        from_node_id=op_name_to_node_id_map[producer_op.get_friendly_name()],
                        to_node_id=op_name_to_node_id_map[consumer_op.get_friendly_name()],
                        tensor_shape=tensor_shape,
                        input_port_id=input_port_id,
                        output_port_id=output_port_id,
                        dtype=GraphConverter.convert_ov_dtype_to_nncf_dtype(tensor_dtype)
                    )

        return nncf_graph
