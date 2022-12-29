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

from typing import List

import openvino.runtime as ov
from openvino.runtime import opset9 as opset

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.openvino_native.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVOutputInsertionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVModelExtractionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVNodeRemovingCommand


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

    def transform(self, transformation_layout: TransformationLayout) -> ov.Model:
        """
        Applies transformations by type-callback on the model.

        :param transformations: lisf of the TransformationCommand transformations.
        """
        output_insert_transformations = []
        bias_correction_transformations = []
        node_removing_transformations = []
        model_extraction_transformation = None

        transformations = transformation_layout.transformations

        for transformation in transformations:
            if isinstance(transformation, OVOutputInsertionCommand):
                output_insert_transformations.append(transformation)
            elif isinstance(transformation, OVModelExtractionCommand):
                model_extraction_transformation = transformation
            elif isinstance(transformation, OVBiasCorrectionCommand):
                bias_correction_transformations.append(transformation)
            elif isinstance(transformation, OVNodeRemovingCommand):
                node_removing_transformations.append(transformation)

        if output_insert_transformations:
            self._apply_output_insertion_transformations(output_insert_transformations)
        if bias_correction_transformations:
            self._apply_bias_correction_transformations(bias_correction_transformations)
        if model_extraction_transformation:
            self._model = self._apply_model_extraction_transformation(model_extraction_transformation)
        if node_removing_transformations:
            self._apply_node_removing_transformation(node_removing_transformations)

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

    def _apply_bias_correction_transformations(self, transformations: List[OVBiasCorrectionCommand]) -> None:
        """
        Applies bias correction transformations on the model.
        :param transformations: List of the bias correction transformations.
        """
        for transformation in transformations:
            node_name = transformation.target_point.target_node_name

            biased_node = self.name_to_node_mapping[node_name]
            bias_port_id = transformation.target_point.port_id
            biased_port = biased_node.input(bias_port_id)
            potential_bias = biased_node.input_value(bias_port_id).node

            if potential_bias.get_type_name() == 'Convert':
                biased_port = potential_bias.input(0)
                potential_bias = potential_bias.input_value(0).node
            assert potential_bias.get_type_name() == 'Constant'
            new_bias = opset.constant(transformation.bias_value, dtype=potential_bias.get_element_type())
            biased_port.replace_source_output(new_bias.output(0))

    def _apply_model_extraction_transformation(self, transformation: OVModelExtractionCommand) -> ov.Model:
        """
        Extracts sub-model from the original based on the inputs and outputs names.
        :param transformation: Model extraction transformation.
        :return: Extracted sub-model.
        """
        params, results = [], []
        for input_name in transformation.inputs:
            input_node = self.name_to_node_mapping[input_name]
            input_port = input_node.input(0)
            input_node_output = input_port.get_source_output()
            new_param = opset.parameter(shape=input_node_output.get_shape(),
                                        dtype=input_node_output.get_element_type(),
                                        name=input_name)
            input_port.replace_source_output(new_param.output(0))
            params.append(new_param)

        for output_name in transformation.outputs:
            output_node = self.name_to_node_mapping[output_name]
            for node_out in output_node.outputs():
                results.append(opset.result(node_out, name=output_name))
        
        if not results:
            results = [r.node for r in self._model.outputs]

        return ov.Model(results, params)

    def _apply_node_removing_transformation(self, transformations: List[OVNodeRemovingCommand]) -> None:
        """
        Removes the layers from the model.

        :param transformations: lisf of the node removing transformations.
        """
        for transformation in transformations:
            node = self.name_to_node_mapping[transformation.target_point.target_node_name]

            node_input = node.input(0).get_source_output()
            for node_output in node.outputs():
                for target_in in node_output.get_target_inputs():
                    target_in.replace_source_output(node_input)
