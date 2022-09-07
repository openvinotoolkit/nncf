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
    """
    The class, which helps to prepare the ONNX model for work with Post-Training Algorithms.
    Implements methods for adding necessary information to the ONNX model.
    """
    TARGET_OPSET_VERSION = 13
    TARGET_IR_VERSION = 7

    @staticmethod
    def add_input_from_initializer(model: onnx.ModelProto) -> onnx.ModelProto:
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

        modified_model = deepcopy(model)
        add_const_value_infos_to_graph(modified_model.graph)
        onnx.checker.check_model(modified_model)
        return modified_model

    @staticmethod
    def infer_models_shape(model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Infers 'model' and saves the edges shape into the 'model'.
        """
        # ONNX shape inference
        # https://github.com/onnx/onnx/blob/main/docs/proposals/SymbolicShapeInfProposal.md
        onnx.checker.check_model(model)
        inferred_shape_model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(inferred_shape_model)
        return inferred_shape_model

    @staticmethod
    def convert_opset_version(model: onnx.ModelProto, opset_version: int,
                              ir_version: int) -> onnx.ModelProto:
        """
        Converts 'model' Opset Version to 'opset_version' if possible and changes IR Version to 'ir_version'.
        If the 'model' can not be converted returns the original 'model'.
        """
        # pylint: disable=no-member
        onnx.checker.check_model(model)
        model_opset = model.opset_import[0].version
        model_ir_version = model.ir_version
        nncf_logger.debug('Original Opset Version = {}'.format(model_opset))
        nncf_logger.debug('Original IR Version = {}'.format(model_ir_version))

        modified_model = model
        if model_opset >= opset_version and model_ir_version >= ir_version:
            nncf_logger.info(
                f"The model Opset Version {opset_version} and IR Version are equal or higher. Using the original model")
            return modified_model
        try:
            modified_model = convert_version(modified_model, opset_version)
            onnx.checker.check_model(modified_model)
            nncf_logger.debug(
                'The model was successfully converted  to the Opset Version = {}'.format(
                    modified_model.opset_import[0].version))
        except (RuntimeError, ConvertError):
            nncf_logger.error(
                f"Couldn't convert target model to the Opset Version {opset_version}. Using the original model")
            return modified_model

        if model_ir_version < ir_version:
            op = onnx.OperatorSetIdProto()
            op.version = opset_version
            modified_model = onnx.helper.make_model(modified_model.graph, ir_version=ir_version, opset_imports=[op])
            onnx.checker.check_model(modified_model)
            nncf_logger.debug(
                'The model was successfully converted  to the Opset Version = {}'.format(
                    modified_model.opset_import[0].version))
            return modified_model

        return modified_model

    @staticmethod
    def replace_empty_node_name(model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Sets an unique name to every empty node in 'model'.
        NNCFGraph expects every node to have an unique name.
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

    @staticmethod
    def normalize_model(model: onnx.ModelProto, convert_opset_version: bool = True) -> onnx.ModelProto:
        """
        Takes the 'model' and cooks it for the Post-Training Algorithms.
        """
        nncf_logger.info('Preparing the model for the Post-Training Algorithms.')
        modified_model = deepcopy(model)
        modified_model = ONNXModelNormalizer.infer_models_shape(modified_model)
        # TODO(kshpv): probably add_input_from_initializer() should be removed with the higher version of onnx package.
        modified_model = ONNXModelNormalizer.add_input_from_initializer(modified_model)
        modified_model = ONNXModelNormalizer.replace_empty_node_name(modified_model)
        if convert_opset_version:
            modified_model = ONNXModelNormalizer.convert_opset_version(modified_model,
                                                                       ONNXModelNormalizer.TARGET_OPSET_VERSION,
                                                                       ONNXModelNormalizer.TARGET_IR_VERSION)
        nncf_logger.info('The model was successfully processed.')
        return modified_model
