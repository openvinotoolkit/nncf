"""
 Copyright (c) 2023 Intel Corporation
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

from typing import List
from enum import Enum

import openvino.runtime as ov
import numpy as np
from openvino.runtime import opset9 as opset

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.openvino_native.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVOutputInsertionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVFQNodeRemovingCommand


class ModelPrecision(Enum):
    """
    Describes the model precision based on the precision of floating point constants.

    :param FP32:
    :param FP16:
    """

    FP32 = 'FP32'
    FP16 = 'FP16'


class OVModelTransformer(ModelTransformer):
    """
    Applies transformations to an OpenVINO model.
    """

    def __init__(self, model: ov.Model):
        """
        Initializes Model Transformer.

        :param model: OpenVINO model to be transformed.
        """
        super().__init__(model)
        self._model = model.clone()
        self.name_to_node_mapping = {op.get_friendly_name(): op for op in self._model.get_ops()}
        self._model_precision = ModelPrecision.FP32
        for op in self._model.get_ops():
            if op.get_type_name() == 'Constant':
                if op.get_element_type().is_real():
                    if op.get_element_type() == ov.Type(np.float16):
                        self._model_precision = ModelPrecision.FP16
                        break


    def transform(self, transformation_layout: TransformationLayout) -> ov.Model:
        """
        Applies transformations by type-callback on the model.

        :param transformations: lisf of the TransformationCommand transformations.
        """
        output_insertion_transformations = []
        fq_nodes_removing_transformations = []
        quantizer_insertion_transformations = []
        transformations = transformation_layout.transformations

        for transformation in transformations:
            if isinstance(transformation, OVOutputInsertionCommand):
                output_insertion_transformations.append(transformation)
            elif isinstance(transformation, OVFQNodeRemovingCommand):
                fq_nodes_removing_transformations.append(transformation)
            elif isinstance(transformation, OVQuantizerInsertionCommand):
                quantizer_insertion_transformations.append(transformation)

        if output_insertion_transformations:
            self._apply_output_insertion_transformations(output_insertion_transformations)
        if fq_nodes_removing_transformations:
            self._apply_fq_nodes_removing_transformation(fq_nodes_removing_transformations)
        if quantizer_insertion_transformations:
            self._apply_quantizer_insertion_transformations(quantizer_insertion_transformations)

        return self._model

    def _apply_output_insertion_transformations(self, transformations: List[OVOutputInsertionCommand]) -> None:
        """
        Applies incoming transformations to the model.

        :param transformations: list of the OVOutputInsertionCommand transformations.
        """
        extra_model_outputs = self._get_extra_model_outputs(transformations)
        self._model = self._insert_outputs(self._model, outputs=extra_model_outputs)

    def _get_extra_model_outputs(self,
                                 transformations: List[OVOutputInsertionCommand]) -> List[ov.Output]:
        """
        Collects extra model outputs based on transformations.

        :param transformations: lisf of the OVOutputInsertionCommand.
        :return: list of the output names.
        """
        extra_model_outputs = []
        for transformation in transformations:
            node_name = transformation.target_point.target_node_name
            node = self.name_to_node_mapping[node_name]
            port_id = transformation.target_point.port_id
            if transformation.target_point.type == TargetType.POST_LAYER_OPERATION:
                output = node.output(port_id)
                extra_model_outputs.append(output)
            elif transformation.target_point.type == TargetType.PRE_LAYER_OPERATION:
                output = node.input_value(port_id)
                extra_model_outputs.append(output)
            else:
                raise NotImplementedError(f'Unsupported target point type {transformation.target_point.type}')

        return extra_model_outputs

    @staticmethod
    def _insert_outputs(model: ov.Model, outputs: List[ov.Output]) -> ov.Model:
        """
        Takes a model and adds outputs based on the list of ov.Output.

        :param model: OpenVINO model.
        :param outputs: list of ov.Output.
        :return: modified model.
        """
        model_outputs = model.get_results()
        params = model.get_parameters()
        extra_model_outputs = []
        for output in outputs:
            output_name = output.get_node().get_friendly_name()
            port_id = output.get_index()
            result = opset.result(output, name=f'Result_{output_name}.{port_id}')
            extra_model_outputs.append(result)

        return ov.Model(model_outputs + extra_model_outputs, params)

    def _apply_fq_nodes_removing_transformation(self, transformations: List[OVFQNodeRemovingCommand]) -> None:
        """
        Removes the layers from the model.
        :param transformations: lisf of the node removing transformations.
        """
        for transformation in transformations:
            node = self.name_to_node_mapping[transformation.target_point.target_node_name]

            node_input = node.input_value(0)
            for node_output in node.outputs():
                for target_in in node_output.get_target_inputs():
                    target_in.replace_source_output(node_input)
            del self.name_to_node_mapping[transformation.target_point.target_node_name]

    def _apply_quantizer_insertion_transformations(
            self,
            transformations: List[OVQuantizerInsertionCommand]) -> None:
        """
        Applies transformations on the model.

        :param transformations: List of the OVQuantizerInsertionCommand transformations.
        """
        for transformation in transformations:
            self._insert_fake_quantize_op(transformation)

    def _insert_fake_quantize_op(self, transformation: OVQuantizerInsertionCommand) -> None:
        fq_params = transformation.quantizer_parameters
        input_low = fq_params.input_low
        input_high = fq_params.input_high
        output_low = fq_params.output_low
        output_high = fq_params.output_high
        levels = fq_params.levels

        def _convert_to_fp16(data):
            return opset.convert(data.astype(np.float16), np.float32)

        if self._model_precision == ModelPrecision.FP16:
            input_low = _convert_to_fp16(input_low)
            input_high = _convert_to_fp16(input_high)
            output_low = _convert_to_fp16(output_low)
            output_high = _convert_to_fp16(output_high)

        node_name = transformation.target_point.target_node_name
        target_node = self.name_to_node_mapping[node_name]
        port_id = transformation.target_point.port_id
        transform_type = transformation.target_point.type
        if transform_type in [TargetType.PRE_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]:
            inp_node = target_node.input(port_id)
            input_node_output = inp_node.get_source_output()
            name = 'fq_weights' if transform_type == TargetType.OPERATION_WITH_WEIGHTS else 'fq_input'
            fq_name = f'{node_name}/{name}_{port_id}'
            fq = opset.fake_quantize(input_node_output, input_low, input_high,
                                     output_low, output_high, levels, name=fq_name)
            inp_node.replace_source_output(fq.output(0))
        elif transform_type == TargetType.POST_LAYER_OPERATION:
            output = target_node.output(port_id)
            target_inputs = output.get_target_inputs()
            fq_name = f'{node_name}/fq_output_{port_id}'
            fq = opset.fake_quantize(output, input_low, input_high,
                                     output_low, output_high, levels, name=fq_name)
            for inp_node in target_inputs:
                inp_node.replace_source_output(fq.output(0))
        else:
            raise RuntimeError(f'Incorrect target point type {transform_type}')
