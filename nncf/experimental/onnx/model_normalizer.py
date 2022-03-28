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

import onnx
from onnx import version_converter  # pylint: disable=no-name-in-module

from nncf.common.utils.logger import logger as nncf_logger


class ONNNXModelNormalizer:
    @staticmethod
    def modify_onnx_model_for_quantization(model: onnx.ModelProto) -> onnx.ModelProto:
        # pylint: disable=no-member
        def add_input_from_initializer(model: onnx.ModelProto) -> None:
            """
            Currently onnx.shape_inference doesn't use the shape of initializers, so add
            that info explicitly as ValueInfoProtos.
            Mutates the model.
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

        onnx.checker.check_model(model)
        nncf_logger.info('Original opset = {}'.format(model.opset_import[0].version))
        nncf_logger.info('Original ir_version = {}'.format(model.ir_version))

        model.ir_version = 7  # Due to the 'Shufflenet-v1
        add_input_from_initializer(model)
        infered_model = onnx.shape_inference.infer_shapes(model)
        modified_model = version_converter.convert_version(infered_model, 13)

        onnx.checker.check_model(modified_model)
        nncf_logger.info(
            'Successfully converted the model to the opset = {}'.format(modified_model.opset_import[0].version))

        for i, node in enumerate(modified_model.graph.node):
            if node.name == '':
                node.name = node.op_type + '_nncf_' + str(i)
        return modified_model
