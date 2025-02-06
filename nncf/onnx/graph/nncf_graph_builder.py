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
from collections import Counter
from typing import Any, Dict, List, Optional, Set

import onnx

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.onnx.graph.metatypes.groups import CONSTANT_WEIGHT_LAYER_METATYPES
from nncf.onnx.graph.metatypes.groups import OPERATIONS_WITH_BIAS
from nncf.onnx.graph.metatypes.groups import POSSIBLE_WEIGHT_LAYER_METATYPES
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXGemmMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXOpMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXOpWithWeightsMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import get_metatype
from nncf.onnx.graph.metatypes.onnx_metatypes import get_tensor_edge_name
from nncf.onnx.graph.onnx_helper import get_children_node_mapping
from nncf.onnx.graph.onnx_helper import get_edge_dtype
from nncf.onnx.graph.onnx_helper import get_edge_info_mapping
from nncf.onnx.graph.onnx_helper import get_edge_shape
from nncf.onnx.graph.onnx_helper import get_input_port_id_for_node_after_input
from nncf.onnx.graph.onnx_helper import get_model_inputs
from nncf.onnx.graph.onnx_helper import get_output_port_id_for_node_before_output
from nncf.onnx.graph.onnx_helper import get_parents_node_mapping
from nncf.onnx.graph.onnx_helper import is_node_has_shared_weight


class ONNXLayerAttributes(BaseLayerAttributes):
    """
    Every NNCFNode for ONNX backend has a ONNXLayerAttributes.
    If node has weight tensor(-s), information for algorithms about weight is stored in weight_attrs.
    If node has bias tensor, information for algorithms about bias is stored in bias_attrs.
    If node has attributes needed for algorithms, they are stored in node_attrs.
    E.g. 'transA' attribute of Gemm node for Quantization.
    """

    def __init__(
        self,
        weight_attrs: Optional[Dict[int, Dict]] = None,
        bias_attrs: Optional[Dict[str, Any]] = None,
        node_attrs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param weight_attrs: Maps input port id associated with weight to a weight description.
        :param bias_attrs: Maps bias tensor name associated with weight to a weight description.
        :param node_attrs: Maps attribute name to an attribute value.
        """
        self.weight_attrs = weight_attrs if weight_attrs is not None else {}
        self.bias_attrs = bias_attrs if bias_attrs is not None else {}
        self.node_attrs = node_attrs if node_attrs is not None else {}

    def has_weight(self) -> bool:
        return bool(self.weight_attrs)

    def has_bias(self) -> bool:
        return bool(self.bias_attrs)

    def has_node_attrs(self) -> bool:
        return bool(self.node_attrs)


def get_constant_weight_port_ids(metatype: ONNXOpMetatype) -> List[int]:
    """
    Returns port ids on which metatype must have a weight based on Operation definition.

    :param metatype: Metatype.
    :return: Port ids.
    """
    if metatype in CONSTANT_WEIGHT_LAYER_METATYPES:
        return metatype.weight_port_ids
    return []


def get_possible_weight_port_ids(metatype: ONNXOpMetatype) -> List[int]:
    """
    Returns weight port ids on which metatype could have a weight.
    Example: ONNXMatMulMetatype could have activations or weights on input port ids: 0, 1

    :param metatype: Metatype.
    :return: Port ids.
    """
    if metatype in POSSIBLE_WEIGHT_LAYER_METATYPES:
        return metatype.possible_weight_ports
    return []


def get_bias_tensor_port_id(metatype: ONNXOpWithWeightsMetatype) -> Optional[int]:
    """
    Returns input port id, where a bias tensor should output.

    :param node: Node, for which input port id is returned,
    :return: Input port id, where a weight bias should output or None if node can not have bias.
    """
    if metatype in OPERATIONS_WITH_BIAS:
        return metatype.bias_port_id
    return None


def _get_weight_port_ids(
    node: onnx.NodeProto,
    model: onnx.ModelProto,
    parents_node_mapping: Dict[str, onnx.NodeProto],
) -> Set[int]:
    """
    Returns all weight input ports.
    First, add constant weight port ids from metatype.
    Second, add weight port ids determined dynamically if metatype could have them.

    :param node: ONNX node.
    :param model: ONNX model.
    :param parents_node_mapping: Mapping from edge name to node which outputs this edge.
    :return: Port ids with weights.
    """
    port_ids = set()
    metatype = get_metatype(model, node)
    constant_port_ids = get_constant_weight_port_ids(metatype)
    port_ids.update(constant_port_ids)
    possible_port_ids = get_possible_weight_port_ids(metatype)
    for port_id in possible_port_ids:
        if get_tensor_edge_name(model, node, port_id, parents_node_mapping):
            port_ids.add(port_id)
    return port_ids


def _is_node_with_bias(node: onnx.NodeProto, model: onnx.ModelProto) -> bool:
    """
    Returns True if node has bias tensor, otherwise - False.

    :param node: ONNX node.
    :param model: ONNX model.
    :return: True if node has bias tensor, otherwise - False.
    """
    metatype = get_metatype(model, node)
    bias_tensor_port_id = get_bias_tensor_port_id(metatype)
    if bias_tensor_port_id is not None and len(node.input) > bias_tensor_port_id:
        return True
    return False


def _get_gemm_attrs(node: onnx.NodeProto) -> Dict[str, int]:
    """
    Returns transpose attributes of GEMM node.

    :param node: GEMM node.
    :return: Transpose attributes.
    """
    gemm_attrs = {"transA": 0, "transB": 0}
    attribute_names = ["transA", "transB"]
    for attr in node.attribute:
        if attr.name in attribute_names:
            gemm_attrs[attr.name] = onnx.helper.get_attribute_value(attr)
    return gemm_attrs


def _get_node_attrs(node: onnx.NodeProto, model: onnx.ModelProto) -> Dict[str, Any]:
    """
    Returns node attributes.

    :param node: Node.
    :param model: ONNX model.
    :return : Node attributes.
    """
    metatype = get_metatype(model, node)
    if metatype == ONNXGemmMetatype:
        return _get_gemm_attrs(node)
    return {}


def _get_bias_attr(
    node: onnx.NodeProto,
    model: onnx.ModelProto,
    parents_node_mapping: Dict[str, onnx.NodeProto],
) -> Dict[str, str]:
    """
    Returns bias tensor attributes.

    :param node: ONNX node.
    :param model: ONNX model.
    :param parents_node_mapping: Mapping from edge name to node which outputs this edge.
    :return: Bias tensor attributes.
    """
    bias_attrs = {}
    metatype = get_metatype(model, node)
    if _is_node_with_bias(node, model):
        bias_tensor_port_id = get_bias_tensor_port_id(metatype)
        bias_edge_name = get_tensor_edge_name(model, node, bias_tensor_port_id, parents_node_mapping)
        bias_attrs["name"] = bias_edge_name
    return bias_attrs


class GraphConverter:
    """
    Builds the NNCFGraph from an ONNX model.
    """

    @staticmethod
    def _replace_empty_node_name(model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Sets a unique name to every node in 'model' with empty name field.
        NNCFGraph expects every node to have a unique name.

        :param model: ONNX model.
        :return: ONNX model with filled nodes.
        """
        for i, node in enumerate(model.graph.node):
            if node.name == "":
                node.name = node.op_type + "_nncf_" + str(i)

        name_counter = Counter([node.name for node in model.graph.node])

        if max(name_counter.values()) > 1:
            raise nncf.ValidationError(
                f"Nodes {[(name, cnt) for name, cnt in name_counter.items() if cnt > 1]} "
                "(name, counts) occurred more than once. "
                "NNCF expects every node to have a unique name."
            )

        return model

    @staticmethod
    def _add_nncf_input_nodes(
        model: onnx.ModelProto,
        nncf_graph: NNCFGraph,
        edge_info_mapping: Dict[str, onnx.ValueInfoProto],
        children_node_mapping: Dict[str, List[onnx.NodeProto]],
    ) -> None:
        """
        Adds special NNCF Input nodes to NNCFGraph.
        For all the ONNX model inputs, the special NNCF Input node is placed and then corresponding edges are added.
        :param model: ONNX model.
        :param nncf_graph: NNCFGraph, in which the new nodes will be added.
        :param edge_info_mapping: Mapping from edge name to the edge info.
        :param children_node_mapping: Mapping from edge name to nodes which consume this edge as an input.
        :return: None.
        """
        for i, _input in enumerate(get_model_inputs(model)):
            input_name = _input.name
            layer_attributes = ONNXLayerAttributes()
            input_node = nncf_graph.add_nncf_node(
                node_name=MODEL_INPUT_OP_NAME + "_" + str(i),
                node_type=NNCFGraphNodeType.INPUT_NODE,
                node_metatype=InputNoopMetatype,
                layer_attributes=layer_attributes,
            )
            to_nodes = children_node_mapping[input_name]

            input_node_node_id = input_node.node_id
            edge = edge_info_mapping[input_name]
            input_shape = get_edge_shape(edge)
            onnx_dtype = get_edge_dtype(edge)
            nncf_dtype = GraphConverter.convert_onnx_dtype_to_nncf_dtype(onnx_dtype)
            output_port_id = 0

            for node in to_nodes:
                to_node_id = nncf_graph.get_node_by_name(node.name).node_id
                input_port_id = get_input_port_id_for_node_after_input(input_name, node)
                nncf_graph.add_edge_between_nncf_nodes(
                    from_node_id=input_node_node_id,
                    to_node_id=to_node_id,
                    tensor_shape=input_shape,
                    input_port_id=input_port_id,
                    output_port_id=output_port_id,
                    dtype=nncf_dtype,
                )

    @staticmethod
    def _add_nncf_output_nodes(
        model: onnx.ModelProto,
        nncf_graph: NNCFGraph,
        edge_info_mapping: Dict[str, onnx.ValueInfoProto],
        parents_node_mapping: Dict[str, onnx.NodeProto],
    ) -> None:
        """
        Adds special NNCF Output nodes to NNCFGraph.
        For all the ONNX model outputs, the special NNCF Output node is placed and then corresponding edges are added.
        :param model: ONNX model.
        :param nncf_graph: NNCFGraph, in which the new nodes will be added.
        :param edge_info_mapping: Mapping from edge name to the edge info.
        :param parents_node_mapping: Mapping from edge name to node which outputs this edge.
        :return: None.
        """
        for i, _output in enumerate(model.graph.output):
            output_name = _output.name
            if output_name in parents_node_mapping:
                layer_attributes = ONNXLayerAttributes()
                output_node = nncf_graph.add_nncf_node(
                    node_name=MODEL_OUTPUT_OP_NAME + "_" + str(i),
                    node_type=NNCFGraphNodeType.OUTPUT_NODE,
                    node_metatype=OutputNoopMetatype,
                    layer_attributes=layer_attributes,
                )

                from_node = parents_node_mapping[output_name]

                output_node_node_id = output_node.node_id
                edge = edge_info_mapping[output_name]
                output_shape = get_edge_shape(edge)
                onnx_dtype = get_edge_dtype(edge)
                nncf_dtype = GraphConverter.convert_onnx_dtype_to_nncf_dtype(onnx_dtype)
                input_port_id = 0
                from_node_id = nncf_graph.get_node_by_name(from_node.name).node_id
                output_port_id = get_output_port_id_for_node_before_output(output_name, from_node)
                nncf_graph.add_edge_between_nncf_nodes(
                    from_node_id=from_node_id,
                    to_node_id=output_node_node_id,
                    tensor_shape=output_shape,
                    input_port_id=input_port_id,
                    output_port_id=output_port_id,
                    dtype=nncf_dtype,
                )

    @staticmethod
    def convert_onnx_dtype_to_nncf_dtype(onnx_dtype: int) -> Dtype:
        """
        Converts the data type from the ONNX domain to the NNCF domain.

        :param np_dtype: ONNX data type.
        :return: NNCF data type.
        """
        return Dtype.FLOAT if onnx_dtype == int(onnx.TensorProto.FLOAT) else Dtype.INTEGER

    @staticmethod
    def create_nncf_graph(onnx_model: onnx.ModelProto) -> NNCFGraph:
        """
        Creates NNCFGraph from 'onnx_model'.
        Initially, ONNXGraph is built. All nodes from onnx_model which have valid metatype are added to NNCFGraph.
        Then, corresponding edges are added to the NNCFGraph with shape, type, output and input port ids.
        In the last step, special NNCF Input and Output nodes are added.
        :param onnx_model: ONNX model.
        :return: NNCFGraph.
        """
        onnx_model = GraphConverter._replace_empty_node_name(onnx_model)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        edge_info_mapping = get_edge_info_mapping(onnx_model)
        children_node_mapping = get_children_node_mapping(onnx_model)
        parents_node_mapping = get_parents_node_mapping(onnx_model)
        nncf_graph = NNCFGraph()
        for node in onnx_model.graph.node:
            metatype = get_metatype(onnx_model, node)
            weight_port_ids = _get_weight_port_ids(node, onnx_model, parents_node_mapping)
            is_shared = None
            weight_attrs = {}
            node_attrs = _get_node_attrs(node, onnx_model)
            bias_attrs = _get_bias_attr(node, onnx_model, parents_node_mapping)
            if weight_port_ids:  # If node has weight
                weight_edge_names = []
                for weight_port_id in weight_port_ids:
                    weight_edge_name = node.input[weight_port_id]
                    weight_edge_names.append(weight_edge_name)
                    edge = edge_info_mapping[weight_edge_name]
                    weight_shape = get_edge_shape(edge)
                    weight_attrs[weight_port_id] = {"name": weight_edge_name, "shape": weight_shape}
                    if not is_shared and is_node_has_shared_weight(node, weight_port_id, children_node_mapping):
                        is_shared = True

            layer_attributes = ONNXLayerAttributes(
                weight_attrs=weight_attrs, bias_attrs=bias_attrs, node_attrs=node_attrs
            )
            nncf_graph.add_nncf_node(
                node_name=node.name,
                node_type=node.op_type,
                node_metatype=metatype,
                layer_attributes=layer_attributes,
                is_shared=is_shared,
            )

        for node in onnx_model.graph.node:
            for output_port_id, output_edge_name in enumerate(node.output):
                for consumed_node in children_node_mapping[output_edge_name]:
                    edge = edge_info_mapping.get(output_edge_name)
                    if edge is None:
                        # If the edge is None it means that the edge was not added during shape inference of ONNX model.
                        # BatchNorm exported in Training mode has unused outputs edges:
                        # mean, var, saved_mean, saved_var. NNCFGraph should not contain such edges.
                        continue
                    input_port_id = get_input_port_id_for_node_after_input(output_edge_name, consumed_node)

                    in_node_id = nncf_graph.get_node_by_name(node.name).node_id
                    output_node_id = nncf_graph.get_node_by_name(consumed_node.name).node_id
                    tensor_shape = get_edge_shape(edge)
                    onnx_dtype = get_edge_dtype(edge)
                    nncf_dtype = GraphConverter.convert_onnx_dtype_to_nncf_dtype(onnx_dtype)

                    nncf_graph.add_edge_between_nncf_nodes(
                        from_node_id=in_node_id,
                        to_node_id=output_node_id,
                        tensor_shape=tensor_shape,
                        input_port_id=input_port_id,
                        output_port_id=output_port_id,
                        dtype=nncf_dtype,
                    )

        GraphConverter._add_nncf_input_nodes(onnx_model, nncf_graph, edge_info_mapping, children_node_mapping)
        GraphConverter._add_nncf_output_nodes(onnx_model, nncf_graph, edge_info_mapping, parents_node_mapping)
        return nncf_graph
