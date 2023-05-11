# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from collections import deque
from typing import Callable, Dict, List, Tuple

import numpy as np
import openvino.runtime as ov
from openvino._pyopenvino import DescriptorTensor  # pylint: disable=no-name-in-module
from openvino.runtime import opset9 as opset

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.model_transformer import TModel
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.openvino.graph.node_utils import get_result_node_name
from nncf.openvino.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.openvino.graph.transformations.commands import OVFQNodeRemovingCommand
from nncf.openvino.graph.transformations.commands import OVInplaceFnInsertionCommand
from nncf.openvino.graph.transformations.commands import OVModelExtractionCommand
from nncf.openvino.graph.transformations.commands import OVOutputInsertionCommand
from nncf.openvino.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.openvino.graph.transformations.commands import OVWeightUpdateCommand
from nncf.quantization.fake_quantize import FakeQuantizeParameters


class OVModelTransformer(ModelTransformer):
    """
    Applies transformations to an OpenVINO model.
    """

    def __init__(self, model: TModel):
        super().__init__(model)
        self._command_transformation_ordered_pairs = [
            (OVFQNodeRemovingCommand, self._apply_fq_nodes_removing_transformation),
            (OVQuantizerInsertionCommand, self._apply_quantizer_insertion_transformations),
            (OVBiasCorrectionCommand, self._apply_bias_correction_transformations),
            (OVWeightUpdateCommand, self._apply_weight_update_transformations),
            (OVModelExtractionCommand, self._apply_model_extraction_transformation),
            (OVInplaceFnInsertionCommand, self._apply_insert_operation),
            (OVOutputInsertionCommand, self._apply_output_insertion_transformations),
        ]

    @staticmethod
    def _get_name_to_node_mapping(model: ov.Model) -> Dict[str, ov.Node]:
        """
        Returns name to node mapping.

        :param model: Model to get mapping.
        :return: Mapping from node name to node.
        """
        return {op.get_friendly_name(): op for op in model.get_ops()}

    @staticmethod
    def _update_tensor_name(tensors: List[DescriptorTensor], name: str) -> None:
        """
        Updates tensors names in-place.

        :param model: List of the tensors.
        :param name: New name for tensor.
        """
        for tensor in tensors:
            current_names = tensor.get_names()
            current_names.add(name)
            tensor.set_names(current_names)

    def transform(self, transformation_layout: TransformationLayout) -> ov.Model:
        """
        Applies transformations to the model using an out-of-place approach.
        The transformations do not affect the original model, and a new model
        is returned with the transformations applied. If there are no transformations,
        returns a new instance of the original model.

        :param transformation_layout: Transformation commands.
        :return: The new instance of a model with applied transformations.
        """

        transformations = transformation_layout.transformations
        aggregated_transformations = defaultdict(list)
        for transformation in transformations:
            aggregated_transformations[transformation.__class__].append(transformation)

        model = self._model.clone()
        # Inplace transformations; Using deepcopy of model
        for transformation_cls, transformation_fn in self._command_transformation_ordered_pairs:
            transformations = aggregated_transformations[transformation_cls]
            if transformations:
                model = transformation_fn(model, transformations)

        return model

    @staticmethod
    def _apply_output_insertion_transformations(
        model: ov.Model, transformations: List[OVOutputInsertionCommand]
    ) -> ov.Model:
        """
        Applies incoming transformations to the model.

        :param model: Model to apply transformations.
        :param transformations: OVOutputInsertionCommand transformations.
        :return: Model with inserted outputs.
        """
        extra_model_outputs = OVModelTransformer._get_extra_model_outputs(model, transformations)
        return OVModelTransformer._insert_outputs(model, outputs=extra_model_outputs)

    @staticmethod
    def _get_extra_model_outputs(
        model: ov.Model, transformations: List[OVOutputInsertionCommand]
    ) -> List[Tuple[ov.Output, int]]:
        """
        Collects extra model outputs based on transformations.

        :param transformations: lisf of the OVOutputInsertionCommand.
        :return: list of tuples with ov.Output & port_id.
        """
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        extra_model_outputs = []
        for transformation in transformations:
            node_name = transformation.target_point.target_node_name
            node = name_to_node_mapping[node_name]
            port_id = transformation.target_point.port_id
            if transformation.target_point.type == TargetType.POST_LAYER_OPERATION:
                output = node.output(port_id)
                extra_model_outputs.append((output, port_id))
            elif transformation.target_point.type in [
                TargetType.PRE_LAYER_OPERATION,
                TargetType.OPERATION_WITH_WEIGHTS,
            ]:
                output = node.input_value(port_id)
                extra_model_outputs.append((output, output.get_index()))
            else:
                raise NotImplementedError(f"Unsupported target point type {transformation.target_point.type}")

        return extra_model_outputs

    @staticmethod
    def _insert_outputs(model: ov.Model, outputs: List[Tuple[ov.Output, int, Callable[[str, int], str]]]) -> ov.Model:
        """
        Takes a model and adds outputs based on the list of ov.Output.

        :param model: OpenVINO model.
        :param outputs: list of tuples with ov.Output & port_id.
        :return: Model with new outputs.
        """
        results = model.get_results()
        params = model.get_parameters()

        assign_ops = [op for op in model.get_ops() if op.get_type_name() == "Assign"]

        extra_model_outputs = []
        for output, port_id in outputs:
            output_name = output.get_node().get_friendly_name()
            # TODO: (KodiaqQ) check out the models with the Split
            result_name = get_result_node_name(output_name, port_id)
            result = opset.result(output, name=result_name)
            OVModelTransformer._update_tensor_name([result.get_output_tensor(0)], result_name)
            extra_model_outputs.append(result)

        return ov.Model(
            results=results + extra_model_outputs, sinks=assign_ops, parameters=params, name=model.friendly_name
        )

    @staticmethod
    def _apply_fq_nodes_removing_transformation(
        model: ov.Model, transformations: List[OVFQNodeRemovingCommand]
    ) -> ov.Model:
        """
        Removes the layers from the model.

        :param model: Model to apply transformations.
        :param transformations: Node removing transformations.
        :return: Model with removed FakeQuantize nodes.
        """
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        for transformation in transformations:
            node = name_to_node_mapping[transformation.target_point.target_node_name]

            node_input = node.input_value(0)
            for node_output in node.outputs():
                for target_in in node_output.get_target_inputs():
                    target_in.replace_source_output(node_input)
            del name_to_node_mapping[transformation.target_point.target_node_name]
        return model

    @staticmethod
    def _apply_quantizer_insertion_transformations(
        model: ov.Model, transformations: List[OVQuantizerInsertionCommand]
    ) -> ov.Model:
        """
        Applies transformations on the model.

        :param model: Model to apply transformations.
        :param transformations: List of the OVQuantizerInsertionCommand transformations.
        :return: Model with inserted FakeQuantize nodes.
        """
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        for transformation in transformations:
            OVModelTransformer._insert_fake_quantize_op(transformation, name_to_node_mapping)
        return model

    @staticmethod
    def convert_params_to_fp16(
        fq_params: FakeQuantizeParameters,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts FakeQuantize parameters to FP16 precision.

        :param fq_params: FakeQuantize node attributes.
        :return: FakeQuantize parameters in FP16 precision.
        """

        def _convert_to_fp16(data):
            clip_data = np.clip(data, np.finfo(np.float16).min, np.finfo(np.float16).max)
            return clip_data.astype(np.float16)

        input_low = _convert_to_fp16(fq_params.input_low)
        input_high = _convert_to_fp16(fq_params.input_high)
        output_low = _convert_to_fp16(fq_params.output_low)
        output_high = _convert_to_fp16(fq_params.output_high)
        return input_low, input_high, output_low, output_high

    @staticmethod
    def _insert_fake_quantize_op(
        transformation: OVQuantizerInsertionCommand, name_to_node_mapping: Dict[str, ov.Node]
    ) -> None:
        """
        Inserts FakeQuantize Operation to a model which name_to_node_mapping is passed.

        :param transformation: FakeQuantize insertion command.
        :param name_to_node_mapping: Mapping from node name to node instance.
        """
        fq_params = transformation.quantizer_parameters
        input_low = fq_params.input_low
        input_high = fq_params.input_high
        output_low = fq_params.output_low
        output_high = fq_params.output_high
        levels = fq_params.levels

        node_name = transformation.target_point.target_node_name
        target_node = name_to_node_mapping[node_name]
        port_id = transformation.target_point.port_id
        transform_type = transformation.target_point.type
        if transform_type in [TargetType.PRE_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]:
            inp_node = target_node.input(port_id)
            input_node_output = inp_node.get_source_output()
            data_type = inp_node.get_element_type()
            if data_type == ov.Type(np.float16):
                input_low, input_high, output_low, output_high = OVModelTransformer.convert_params_to_fp16(fq_params)
            name = "fq_weights" if transform_type == TargetType.OPERATION_WITH_WEIGHTS else "fq_input"
            fq_name = f"{node_name}/{name}_{port_id}"

            fq = None
            if transform_type == TargetType.OPERATION_WITH_WEIGHTS:
                # If the nodes share one weight tensor, we should have only one quantizer on that
                for out in input_node_output.get_target_inputs():
                    if out.get_node().get_type_name() == "FakeQuantize":
                        fq = out.get_node()
            if fq is None:
                fq = opset.fake_quantize(
                    input_node_output, input_low, input_high, output_low, output_high, levels, name=fq_name
                )
            inp_node.replace_source_output(fq.output(0))
        elif transform_type == TargetType.POST_LAYER_OPERATION:
            output = target_node.output(port_id)
            data_type = output.get_element_type()
            if data_type == ov.Type(np.float16):
                input_low, input_high, output_low, output_high = OVModelTransformer.convert_params_to_fp16(fq_params)
            target_inputs = output.get_target_inputs()
            fq_name = f"{node_name}/fq_output_{port_id}"
            fq = opset.fake_quantize(output, input_low, input_high, output_low, output_high, levels, name=fq_name)
            for inp_node in target_inputs:
                inp_node.replace_source_output(fq.output(0))
        else:
            raise RuntimeError(f"Incorrect target point type {transform_type}")

    @staticmethod
    def _apply_bias_correction_transformations(model, transformations: List[OVBiasCorrectionCommand]) -> ov.Model:
        """
        Applies bias correction transformations on the model.

        :param model: Model to apply transformations.
        :param transformations: List of the bias correction transformations.
        :return: Model with corrected bias.
        """
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        for transformation in transformations:
            node = name_to_node_mapping[transformation.target_point.target_node_name]
            node_inputs = [port.get_node() for port in node.output(0).get_target_inputs()]
            assert any(node.get_type_name() == "Add" for node in node_inputs)

            for node_input in node_inputs:
                if node_input.get_type_name() == "Add":
                    add_node = node_input

            OVModelTransformer._set_const_value(
                add_node, transformation.target_point.port_id, transformation.bias_value
            )
        return model

    @staticmethod
    def _set_const_value(node_with_const: ov.Node, const_port_id: int, const_value: np.ndarray) -> None:
        port = node_with_const.input(const_port_id)
        node = node_with_const.input_value(const_port_id).get_node()

        const_port = None
        const_node = None
        queue = deque([(port, node)])
        while len(queue) != 0:
            curr_port, curr_node = queue.popleft()
            if curr_node.get_type_name() == "Constant":
                const_port = curr_port
                const_node = curr_node
                break
            if len(curr_node.inputs()) == 0:
                break
            queue.append((curr_node.input(0), curr_node.input_value(0).get_node()))

        if const_node is None:
            raise RuntimeError("Constant node was expected but could not find it.")

        const_shape = const_node.get_data().shape
        const_value = np.reshape(const_value, const_shape)
        new_const_node = opset.constant(const_value, dtype=const_node.get_element_type())
        new_const_node.set_friendly_name(const_node.get_friendly_name())
        const_port.replace_source_output(new_const_node.output(0))

    @staticmethod
    def _apply_weight_update_transformations(model, transformations: List[OVWeightUpdateCommand]) -> ov.Model:
        """
        Applies weight update transformation to the model.

        :param transformations: List of the weight update transformations.
        :returns: Transformed model.
        """
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        for transformation in transformations:
            node_with_weight = name_to_node_mapping[transformation.target_point.target_node_name]
            OVModelTransformer._set_const_value(
                node_with_weight, transformation.target_point.port_id, transformation.weight_value  # Weight port id
            )
        return model

    @staticmethod
    def _apply_model_extraction_transformation(
        model: ov.Model, transformations: List[OVModelExtractionCommand]
    ) -> ov.Model:
        """
        Extracts sub-model from the original based on the inputs and outputs names.

        :param model: Model to apply transformations.
        :param transformation: Model extraction transformation.
        :return: Extracted sub-model.
        """
        transformation = transformations[-1]
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        params, results = [], []
        for input_name in transformation.inputs:
            input_node = name_to_node_mapping[input_name]
            if input_name in [tensor.node.get_friendly_name() for tensor in model.inputs]:
                params.append(input_node)
                continue
            input_port = input_node.input(0)
            input_node_output = input_port.get_source_output()
            parameter_name = f"Parameter_{input_name}"
            new_param = opset.parameter(
                shape=input_node_output.partial_shape, dtype=input_node_output.get_element_type(), name=parameter_name
            )
            input_port.replace_source_output(new_param.output(0))
            new_param_tensors = [o.get_tensor() for o in new_param.outputs()]
            OVModelTransformer._update_tensor_name(new_param_tensors, parameter_name)
            params.append(new_param)

        for output_name in transformation.outputs:
            output_node = name_to_node_mapping[output_name]
            for node_out in output_node.outputs():
                result_name = get_result_node_name(output_name, 0)
                new_result = opset.result(node_out, name=result_name)
                OVModelTransformer._update_tensor_name([new_result.get_output_tensor(0)], result_name)
                results.append(new_result)

        if not results:
            results = model.get_results()

        return ov.Model(results, params)

    @staticmethod
    def _apply_insert_operation(model: ov.Model, transformations: OVInplaceFnInsertionCommand) -> ov.Model:
        """
        Applies inplace fn insertion transformation to the model.

        :param transformations: lisf of the OVInplaceFnInsertionCommand.
        :returns: Transformed model.
        """
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        outputs = []
        for transformation in transformations:
            outputs.append(OVModelTransformer._insert_inplace_operation(transformation, name_to_node_mapping))
        return OVModelTransformer._insert_outputs(model, outputs)

    @staticmethod
    def _insert_inplace_operation(
        transformation: OVInplaceFnInsertionCommand, name_to_node_mapping: Dict[str, ov.Node]
    ) -> Tuple[ov.Output, int]:
        """
        Inserts operation inplace to a model which name_to_node_mapping is passed.

        :param transformation: Inplace fn insertion command.
        :param name_to_node_mapping: Mapping from node name to node instance.
        :returns: Pair with inserted node output and corresponded output port id.
        """
        transform_type = transformation.target_point.type

        node_name = transformation.target_point.target_node_name
        target_node = name_to_node_mapping[node_name]
        port_id = transformation.target_point.port_id
        fn_output_port_id = transformation.fn_output_port_id
        if transform_type == TargetType.POST_LAYER_OPERATION:
            new_node = transformation.inplace_op_fn(target_node, port_id)
            return (new_node.output(fn_output_port_id), fn_output_port_id)
        if transform_type in [TargetType.PRE_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]:
            output = target_node.input_value(port_id)
            new_node = transformation.inplace_op_fn(output.get_node(), output.get_index())
            return (new_node.output(fn_output_port_id), fn_output_port_id)
        raise RuntimeError(f"Transform type {transform_type} is not supported")
