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

from collections import Counter
from copy import deepcopy
import onnx
from onnx.version_converter import convert_version, ConvertError  # pylint: disable=no-name-in-module
from nncf.common.utils.logger import logger as nncf_logger


class ONNXModelNormalizer:
    @staticmethod
    def add_input_from_initializer(model: onnx.ModelProto) -> None:
        """
        Currently onnx.shape_inference doesn't use the shape of initializers, so add
        that info explicitly as ValueInfoProtos.
        Mutates the model.

        History of this code
         - After onnx.shape_inference.infer_shapes the model graph value_info doesn't
         include all activations tensors #4102
         - https://github.com/onnx/onnx/issues/4102

        Args:
            model: The ModelProto to update.
        """
        # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
        if model.ir_version < 4:
            nncf_logger.info('Could not process model, as it has {} < 4'.format(model.ir_version))
            return model

        def add_const_value_infos_to_graph(graph: onnx.GraphProto):
            inputs = {i.name for i in graph.input}
            existing_info = {vi.name: vi for vi in graph.input}
            for init in graph.initializer:
                # Check it really is a constant, not an input
                if init.name in inputs:
                    continue

                # The details we want to add
                elem_type = init.data_type
                shape = init.dims

                # Get existing or create new value info for this constant
                vi = existing_info.get(init.name)
                if vi is None:
                    vi = graph.input.add()
                    vi.name = init.name

                # Even though it would be weird, we will not overwrite info even if it doesn't match
                tt = vi.type.tensor_type
                if tt.elem_type == onnx.TensorProto.UNDEFINED:
                    tt.elem_type = elem_type
                if not tt.HasField("shape"):
                    # Ensure we set an empty list if the const is scalar (zero dims)
                    tt.shape.dim.extend([])
                    for dim in shape:
                        tt.shape.dim.add().dim_value = dim

            # Handle subgraphs
            for node in graph.node:
                for attr in node.attribute:
                    # Ref attrs refer to other attrs, so we don't need to do anything
                    if attr.ref_attr_name != "":
                        continue

                    if attr.type == onnx.AttributeProto.GRAPH:
                        add_const_value_infos_to_graph(attr.g)
                    if attr.type == onnx.AttributeProto.GRAPHS:
                        for g in attr.graphs:
                            add_const_value_infos_to_graph(g)

        return add_const_value_infos_to_graph(model.graph)

    @staticmethod
    def convert_opset_version(model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Try to convert model to opset version 13, also adding some important information to the model such shapes.
        """
        # pylint: disable=no-member
        onnx.checker.check_model(model)
        nncf_logger.info('Original opset = {}'.format(model.opset_import[0].version))
        nncf_logger.info('Original ir_version = {}'.format(model.ir_version))

        try:
            modified_model = deepcopy(model)
            modified_model.ir_version = 7  # Due to the 'Shufflenet-v1
            modified_model = convert_version(modified_model, 13)

            # ONNX shape inference
            # https://github.com/onnx/onnx/blob/main/docs/proposals/SymbolicShapeInfProposal.md
            modified_model = onnx.shape_inference.infer_shapes(modified_model)
            ONNXModelNormalizer.add_input_from_initializer(modified_model)

            onnx.checker.check_model(modified_model)

            nncf_logger.info(
                'Successfully converted the model to the opset = {}'.format(modified_model.opset_import[0].version))
        except (RuntimeError, ConvertError):
            modified_model = model
            nncf_logger.error(
                "Couldn't convert target model to opset13. Use original model")

        return modified_model

    @staticmethod
    def replace_empty_node_name(model: onnx.ModelProto):
        """
        NNCFGraph.get_node_by_name() does not allow empty node names.
        NNCF expects every node to have a unique name.
        """
        for i, node in enumerate(model.graph.node):
            if node.name == '':
                node.name = node.op_type + '_nncf_' + str(i)

        name_counter = Counter([node.name for node in model.graph.node])

        if max(name_counter.values()) > 1:
            raise RuntimeError(
                f"Nodes {[(name, cnt) for name, cnt in name_counter.items() if cnt > 1]} "
                "(name, counts) occurred more than once. "
                "NNCF expects every node to have a unique name.")

        return model
