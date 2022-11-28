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
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.utils.logger import logger as nncf_logger

from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OV_OPERATION_METATYPES


class GraphConverter:
    """
    Builds the NNCFGraph from an OpenVINO model.
    """

    @staticmethod
    def _is_valid_openvino_metatype(node: ov.Node) -> bool:
        """
        Checks whether the node has the metatype which should be added to the NNCFGraph.
        :param node: Node to be checked.
        :return: True if the metatype is valid and False if not.
        """
        node_type = node.get_type_name()
        metatype = OV_OPERATION_METATYPES.get_operator_metatype_by_op_name(node_type)
        if metatype == UnknownMetatype:
            node_name = node.get_friendly_name()
            nncf_logger.warning(
                'The node with name {} with type {} was mapped to UnknownMetatype,'
                ' which means that there was not registered such NNCF metatype. '
                'Please, Inform the NNCF developers about this message.'.format(
                    node_name, node_type))
        return True

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

        for node in filter(GraphConverter._is_valid_openvino_metatype, model.get_ops()):
            ov_dtype = node.get_element_type()
            nncf_dtype = GraphConverter.convert_ov_dtype_to_nncf_dtype(ov_dtype)
            node_type = node.get_type_name()
            metatype = OV_OPERATION_METATYPES.get_operator_metatype_by_op_name(node_type)
            nncf_graph.add_nncf_node(node_name=node.get_friendly_name(),
                                     node_type=nncf_dtype,
                                     node_metatype=metatype)

        for op in filter(GraphConverter._is_valid_openvino_metatype, model.get_ops()):
            in_node_id = nncf_graph.get_node_by_name(op.get_friendly_name()).node_id
            for output_port_id, out in enumerate(op.outputs()):
                for inp in out.get_target_inputs():
                    out_node = inp.get_node()
                    if GraphConverter._is_valid_openvino_metatype(out_node):
                        tensor_shape = list(out.shape)
                        output_node_id = nncf_graph.get_node_by_name(out_node.get_friendly_name()).node_id

                        ov_dtype = op.get_element_type().get_type_name()
                        nncf_dtype = GraphConverter.convert_ov_dtype_to_nncf_dtype(ov_dtype)
                        nncf_graph.add_edge_between_nncf_nodes(
                            from_node_id=in_node_id,
                            to_node_id=output_node_id,
                            tensor_shape=tensor_shape,
                            input_port_id=inp.get_index(),
                            output_port_id=output_port_id,
                            dtype=Dtype(nncf_dtype)
                        )

        return nncf_graph
