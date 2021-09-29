import networkx as nx
import onnx

from nncf.common.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype

from nncf.experimental.onnx.graph.helpers import find_nodes_by_input, find_nx_graph_node_by_label, find_output_shape
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ONNX_OPERATION_METATYPES


class GraphConverter:
    @staticmethod
    def create_nncf_graph(onnx_model: onnx.ModelProto):
        digraph = GraphConverter.convert_from_onnx_to_digraph(onnx_model)
        nncf_graph = GraphConverter.convert_from_digraph_to_nncf_graph(digraph)
        return nncf_graph

    @staticmethod
    def convert_from_onnx_to_digraph(onnx_model: onnx.ModelProto) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        node_cnt = 0
        for node in onnx_model.graph.node:
            params = {}
            params["input"] = node.input
            params["output"] = node.output
            params["label"] = node.name
            params["type"] = node.op_type
            params["op_attrs"] = node.attribute
            nx_graph.add_node(node_cnt, **params)
            node_cnt += 1

        inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
        activations_shapes = inferred_model.graph.value_info
        for node, attrs in nx_graph.nodes(data=True):
            outputs = attrs['output']
            for output in outputs:
                nodes = find_nodes_by_input(output, onnx_model.graph)
                shape = find_output_shape(output, activations_shapes)
                for in_node in nodes:
                    nx_graph_in_node = find_nx_graph_node_by_label(in_node.name, nx_graph)
                    nx_graph.add_edge(node, nx_graph_in_node, activation_shape=shape)

        return nx_graph


    @staticmethod
    def convert_from_digraph_to_nncf_graph(digraph):
        nncf_graph = NNCFGraph()
        for node, attrs in digraph.nodes(data=True):
            node_name = attrs['label']
            node_type = attrs['type']
            metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node_type)

            # is_integer_input = False
            # if metatype in INPUT_NOOP_METATYPES and input_infos is not None:
            #     input_id = op_address.call_order
            #     if input_infos[input_id].is_integer_input():
            #         is_integer_input = True
            #
            # is_shared = layer_name in layer_name_vs_node_counts and layer_name_vs_node_counts[layer_name] > 1

            nncf_graph.add_nncf_node(node_name=node_name,
                                     node_type=node_type,
                                     node_metatype=metatype,
                                     layer_attributes=None)
        input_counter = {}
        output_counter = {}
        for u, v, a in digraph.edges(data=True):
            input_counter[v] = input_counter.get(v, -1) + 1
            output_counter[u] = input_counter.get(u, -1) + 1
            nncf_graph.add_edge_between_nncf_nodes(
                from_node_id=u,
                to_node_id=v,
                tensor_shape=a['activation_shape'],
                input_port_id=input_counter[v],
                output_port_id=output_counter[u],
                dtype=Dtype.FLOAT
            )
        return nncf_graph