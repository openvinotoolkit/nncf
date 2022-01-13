import onnx
from google.protobuf.json_format import MessageToDict

from nncf.common.graph import NNCFGraph
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.layer_attributes import Dtype

from nncf.experimental.onnx.graph.onnx_graph import ONNXGraphHelper
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ONNX_OPERATION_METATYPES
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ConstantMetatype


# pylint: disable=no-member

class GraphConverter:
    @staticmethod
    def create_nncf_graph(onnx_model: onnx.ModelProto) -> NNCFGraph:
        nncf_graph = NNCFGraph()
        for node in onnx_model.graph.node:
            node_name = node.name
            node_type = node.op_type
            metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node_type)
            # We don't need to quantize Constants
            if metatype == ConstantMetatype:
                continue
            nncf_graph.add_nncf_node(node_name=node_name,
                                     node_type=node_type,
                                     node_metatype=metatype,
                                     layer_attributes=None)
        input_counter = {}
        output_counter = {}
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
        activations_shapes = inferred_model.graph.value_info
        for output_node in nncf_graph.get_all_nodes():
            output_node_id = output_node.node_id
            outputs = ONNXGraphHelper.get_all_node_outputs(output_node.node_name, onnx_model.graph)
            for output in outputs:
                nodes = ONNXGraphHelper.find_nodes_by_input(output, onnx_model.graph)
                shape = ONNXGraphHelper.find_node_output_shape_in_activation_shapes(output, activations_shapes)
                for in_node in nodes:
                    in_node_id = nncf_graph.get_node_by_name(in_node.name).node_id
                    input_counter[in_node_id] = input_counter.get(in_node_id, -1) + 1
                    output_counter[output_node_id] = input_counter.get(output_node_id, -1) + 1
                    nncf_graph.add_edge_between_nncf_nodes(
                        from_node_id=output_node_id,
                        to_node_id=in_node_id,
                        tensor_shape=shape,
                        input_port_id=input_counter[in_node_id],
                        output_port_id=output_counter[output_node_id],
                        dtype=Dtype.FLOAT
                    )
        # Add Input Nodes
        for i, _input in enumerate(onnx_model.graph.input):
            m_dict = MessageToDict(_input)
            dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
            input_shape = [int(d.get("dimValue")) for d in dim_info]
            input_node = nncf_graph.add_nncf_node(node_name='input_node_' + str(i),
                                                  node_type=NNCFGraphNodeType.INPUT_NODE,
                                                  node_metatype=ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(
                                                      NNCFGraphNodeType.INPUT_NODE),
                                                  layer_attributes=None)
            input_name = _input.name
            to_nodes = ONNXGraphHelper.find_nodes_by_input(input_name, onnx_model.graph)
            for node in to_nodes:
                in_node_id = input_node.node_id
                to_node_id = nncf_graph.get_node_by_name(node.name).node_id
                input_counter[in_node_id] = input_counter.get(input_node.node_id, -1) + 1
                output_counter[to_node_id] = input_counter.get(to_node_id, -1) + 1
                nncf_graph.add_edge_between_nncf_nodes(
                    from_node_id=input_node.node_id,
                    to_node_id=to_node_id,
                    tensor_shape=input_shape,
                    input_port_id=input_counter[in_node_id],
                    output_port_id=output_counter[to_node_id],
                    dtype=Dtype.FLOAT
                )
        # Add Output Nodes
        for i, _output in enumerate(onnx_model.graph.output):
            m_dict = MessageToDict(_output)
            dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
            output_shape = [int(d.get("dimValue")) for d in dim_info]
            output_node = nncf_graph.add_nncf_node(node_name='output_node_' + str(i),
                                                   node_type=NNCFGraphNodeType.OUTPUT_NODE,
                                                   node_metatype=ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(
                                                       NNCFGraphNodeType.OUTPUT_NODE),
                                                   layer_attributes=None)

            output_name = _output.name
            to_nodes = ONNXGraphHelper.find_nodes_by_output(output_name, onnx_model.graph)
            for node in to_nodes:
                out_node_id = output_node.node_id
                to_node_id = nncf_graph.get_node_by_name(node.name).node_id
                input_counter[out_node_id] = input_counter.get(output_node.node_id, -1) + 1
                output_counter[to_node_id] = input_counter.get(to_node_id, -1) + 1
                nncf_graph.add_edge_between_nncf_nodes(
                    from_node_id=to_node_id,
                    to_node_id=output_node.node_id,
                    tensor_shape=output_shape,
                    input_port_id=input_counter[out_node_id],
                    output_port_id=output_counter[to_node_id],
                    dtype=Dtype.FLOAT
                )

        return nncf_graph
