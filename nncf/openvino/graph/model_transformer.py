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

from collections import defaultdict
from collections import deque
from typing import Callable, Dict, List, Tuple

import numpy as np
import openvino.runtime as ov
from openvino._pyopenvino import DescriptorTensor
from openvino.runtime import opset13 as opset

import nncf
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.model_transformer import TModel
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.openvino.graph.model_utils import copy_rt_info
from nncf.openvino.graph.node_utils import get_parameter_node_name
from nncf.openvino.graph.node_utils import get_result_node_name
from nncf.openvino.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.openvino.graph.transformations.commands import OVBiasInsertionCommand
from nncf.openvino.graph.transformations.commands import OVConvertInsertionCommand
from nncf.openvino.graph.transformations.commands import OVExtractIfBodyCommand
from nncf.openvino.graph.transformations.commands import OVFQNodeRemovingCommand
from nncf.openvino.graph.transformations.commands import OVInplaceFnInsertionCommand
from nncf.openvino.graph.transformations.commands import OVModelExtractionCommand
from nncf.openvino.graph.transformations.commands import OVMultiplyInsertionCommand
from nncf.openvino.graph.transformations.commands import OVOutputInsertionCommand
from nncf.openvino.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.openvino.graph.transformations.commands import OVStateLessModelExtractionCommand
from nncf.openvino.graph.transformations.commands import OVUpdateIfBodyCommand
from nncf.openvino.graph.transformations.commands import OVWeightUpdateCommand
from nncf.quantization.fake_quantize import FakeConvertParameters
from nncf.quantization.fake_quantize import FakeQuantizeParameters


class OVModelTransformer(ModelTransformer):
    """
    Applies transformations to an OpenVINO model.
    """

    def __init__(self, model: TModel, inplace: bool = False):
        super().__init__(model)
        self._inplace = inplace
        self._command_transformation_ordered_pairs = [
            (OVFQNodeRemovingCommand, self._apply_fq_nodes_removing_transformation),
            (OVQuantizerInsertionCommand, self._apply_quantizer_insertion_transformations),
            (OVConvertInsertionCommand, self._apply_convert_insertion_transformations),
            (OVBiasCorrectionCommand, self._apply_bias_correction_transformations),
            (OVWeightUpdateCommand, self._apply_weight_update_transformations),
            (OVModelExtractionCommand, self._apply_model_extraction_transformation),
            (OVStateLessModelExtractionCommand, self._apply_stateless_model_extraction_transformation),
            (OVInplaceFnInsertionCommand, self._apply_insert_operation),
            (OVOutputInsertionCommand, self._apply_output_insertion_transformations),
            (OVBiasInsertionCommand, self._apply_bias_insertion_transformations),
            (OVMultiplyInsertionCommand, self._apply_multiply_insertion_transformations),
            (OVUpdateIfBodyCommand, self._apply_update_if_body_transformations),
            (OVExtractIfBodyCommand, self._apply_extract_if_body_transformation),
        ]

    @staticmethod
    def _convert_to_dtype(value, dtype):
        clip_data = np.clip(value, np.finfo(dtype).min, np.finfo(dtype).max)
        return clip_data.astype(dtype)

    @staticmethod
    def _get_name_to_node_mapping(model: ov.Model) -> Dict[str, ov.Node]:
        """
        Returns name to node mapping.

        :param model: Model to get mapping.
        :return: Mapping from node name to node.
        """
        return {op.get_friendly_name(): op for op in model.get_ops()}

    @staticmethod
    def _get_activation_node_names(model: ov.Model) -> List[str]:
        """
        Returns list of the activation node names.

        :param model: Model to get list.
        :return: List with the activation names.
        """
        activation_nodes = set()
        nodes_queue = deque(model.get_parameters())
        while nodes_queue:
            node = nodes_queue.popleft()
            if node.name in activation_nodes:
                continue
            activation_nodes.add(node.name)
            for node_output in node.outputs():
                nodes_queue.extend([i.get_node() for i in node_output.get_target_inputs()])
        return list(activation_nodes)

    @staticmethod
    def _update_tensor_names(tensors: List[DescriptorTensor], names: List[str]) -> None:
        """
        Updates tensors names in-place.

        :param model: List of the tensors.
        :param names: List of the new names for tensors.
        """
        for tensor in tensors:
            current_names = tensor.get_names()
            current_names.update(names)
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

        if self._inplace:
            model = self._model
        else:
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
            output_dtype = transformation.output_dtype
            if transformation.target_point.type == TargetType.POST_LAYER_OPERATION:
                output = node.output(port_id)
                extra_model_outputs.append((output, port_id, output_dtype))
            elif transformation.target_point.type in [
                TargetType.PRE_LAYER_OPERATION,
                TargetType.OPERATION_WITH_WEIGHTS,
            ]:
                output = node.input_value(port_id)
                extra_model_outputs.append((output, output.get_index(), output_dtype))
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

        extra_model_outputs = []
        for output, port_id, dtype in outputs:
            node_output = output
            output_name = node_output.get_node().get_friendly_name()
            result_name = get_result_node_name(output_name, port_id)

            if dtype is not None and node_output.get_element_type() != dtype:
                node_output = opset.convert(output, destination_type=dtype)

            result = opset.result(node_output, name=result_name)
            result_tensor_names = [result_name] + list(output.get_names())
            OVModelTransformer._update_tensor_names([result.get_output_tensor(0)], result_tensor_names)
            extra_model_outputs.append(result)

        model_with_outputs = ov.Model(
            results=results + extra_model_outputs, sinks=model.get_sinks(), parameters=params, name=model.friendly_name
        )
        copy_rt_info(model, model_with_outputs, path=["nncf"])
        return model_with_outputs

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
    def _apply_convert_insertion_transformations(
        model: ov.Model, transformations: List[OVConvertInsertionCommand]
    ) -> ov.Model:
        """
        Applies transformations on the model.

        :param model: Model to apply transformations.
        :param transformations: List of the OVConvertInsertionCommand transformations.
        :return: Model with inserted FakeConvert nodes.
        """
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        for transformation in transformations:
            OVModelTransformer._insert_fake_convert_op(transformation, name_to_node_mapping)
        return model

    @staticmethod
    def _create_constant(value: np.ndarray, dtype: ov.Type, name: str, shared_memory: bool = False) -> ov.Node:
        """
        Creates constant using opset.

        :param value: Numpy value.
        :param type: Constant type.
        :param name: Name for the constant.
        :param shared_memory: Shared memory option.
        :return: ov.Node instance.
        """
        constant_value = value
        # BF16 does not support type conversion in numpy
        if dtype != ov.Type.bf16 and constant_value.dtype != dtype.to_dtype():
            constant_value = OVModelTransformer._convert_to_dtype(constant_value, dtype.to_dtype())

        return opset.constant(constant_value, dtype=dtype, name=name, shared_memory=shared_memory)

    @staticmethod
    def _create_fake_quantize(
        op_output: ov.Output,
        fake_quantize_params: FakeQuantizeParameters,
        fake_quantize_name: str,
        data_type: ov.Type,
    ) -> ov.Node:
        """
        Creates FakeQuantize node.

        :param op_output: Output of the previous node.
        :param fake_quantize_params: FakeQuantizeParameters instance.
        :param fake_quantize_name: New layer name.
        :param data_type: ov.Type instance for data.
        :return: ov.Node instance.
        """

        input_low = fake_quantize_params.input_low.data
        input_high = fake_quantize_params.input_high.data
        output_low = fake_quantize_params.output_low.data
        output_high = fake_quantize_params.output_high.data
        levels = fake_quantize_params.levels

        input_low = OVModelTransformer._create_constant(
            input_low, dtype=data_type, name=f"{fake_quantize_name}/input_low"
        )
        input_high = OVModelTransformer._create_constant(
            input_high, dtype=data_type, name=f"{fake_quantize_name}/input_high"
        )
        output_low = OVModelTransformer._create_constant(
            output_low, dtype=data_type, name=f"{fake_quantize_name}/output_low"
        )
        output_high = OVModelTransformer._create_constant(
            output_high, dtype=data_type, name=f"{fake_quantize_name}/output_high"
        )

        return opset.fake_quantize(
            op_output, input_low, input_high, output_low, output_high, levels, name=fake_quantize_name
        )

    @staticmethod
    def _create_fake_convert(
        op_output: ov.Output,
        fake_convert_params: FakeConvertParameters,
        fake_convert_name: str,
        data_type: ov.Type,
    ) -> ov.Node:
        """
        Creates FakeConvert node.

        :param op_output: Output of the previous node.
        :param fake_convert_params: FakeConvertParameters instance.
        :param fake_convert_name: New layer name.
        :param data_type: ov.Type instance for data.
        :return: ov.Node instance.
        """

        scale = fake_convert_params.scale.data
        shift = fake_convert_params.shift.data

        destination_type = fake_convert_params.destination_type.value
        scale = OVModelTransformer._create_constant(scale, dtype=data_type, name=f"{fake_convert_name}/scale")
        shift = OVModelTransformer._create_constant(shift, dtype=data_type, name=f"{fake_convert_name}/shift")

        return opset.fake_convert(
            data=op_output,
            scale=scale,
            shift=shift,
            destination_type=destination_type,
            name=fake_convert_name,
        )

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

        node_name = transformation.target_point.target_node_name
        target_node = name_to_node_mapping[node_name]
        port_id = transformation.target_point.port_id
        transform_type = transformation.target_point.type
        if transform_type in [TargetType.PRE_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]:
            inp_node = target_node.input(port_id)
            input_node_output = inp_node.get_source_output()
            data_type = inp_node.get_element_type()
            name = "fq_weights" if transform_type == TargetType.OPERATION_WITH_WEIGHTS else "fq_input"
            fq_name = f"{node_name}/{name}_{port_id}"

            fq = None
            if transform_type == TargetType.OPERATION_WITH_WEIGHTS:
                # If the nodes share one weight tensor, we should have only one quantizer on that
                for out in input_node_output.get_target_inputs():
                    if out.get_node().get_type_name() == "FakeQuantize":
                        fq = out.get_node()
            if fq is None:
                fq = OVModelTransformer._create_fake_quantize(
                    op_output=input_node_output,
                    fake_quantize_params=fq_params,
                    fake_quantize_name=fq_name,
                    data_type=data_type,
                )
            inp_node.replace_source_output(fq.output(0))
        elif transform_type == TargetType.POST_LAYER_OPERATION:
            output = target_node.output(port_id)
            data_type = output.get_element_type()
            target_inputs = output.get_target_inputs()
            fq_name = f"{node_name}/fq_output_{port_id}"
            fq = OVModelTransformer._create_fake_quantize(
                op_output=output,
                fake_quantize_params=fq_params,
                fake_quantize_name=fq_name,
                data_type=data_type,
            )
            for inp_node in target_inputs:
                inp_node.replace_source_output(fq.output(0))
        else:
            raise nncf.InternalError(f"Incorrect target point type {transform_type}")

    @staticmethod
    def _insert_fake_convert_op(
        transformation: OVConvertInsertionCommand, name_to_node_mapping: Dict[str, ov.Node]
    ) -> None:
        """
        Inserts FakeConvert Operation to a model which name_to_node_mapping is passed.

        :param transformation: FakeConvert insertion command.
        :param name_to_node_mapping: Mapping from node name to node instance.
        """
        fc_params = transformation.convert_parameters

        node_name = transformation.target_point.target_node_name
        target_node = name_to_node_mapping[node_name]
        port_id = transformation.target_point.port_id
        transform_type = transformation.target_point.type
        name = "weights" if transform_type == TargetType.OPERATION_WITH_WEIGHTS else "input"

        if transform_type in [TargetType.PRE_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]:
            inp_node = target_node.input(port_id)
            input_node_output = inp_node.get_source_output()

            fc = None
            if transform_type == TargetType.OPERATION_WITH_WEIGHTS:
                # If the nodes share one weight tensor, we should have only one quantizer on that
                for out in input_node_output.get_target_inputs():
                    if out.get_node().get_type_name() == "FakeConvert":
                        fc = out.get_node()
            if fc is None:
                data_type = inp_node.get_element_type()
                fc_name = f"{node_name}/fc_{name}_{port_id}"
                fc = OVModelTransformer._create_fake_convert(
                    op_output=input_node_output,
                    fake_convert_params=fc_params,
                    fake_convert_name=fc_name,
                    data_type=data_type,
                )
            inp_node.replace_source_output(fc.output(0))
        elif transform_type == TargetType.POST_LAYER_OPERATION:
            output = target_node.output(port_id)
            data_type = output.get_element_type()
            target_inputs = output.get_target_inputs()
            fc_name = f"{node_name}/fc_output_{port_id}"
            fc = OVModelTransformer._create_fake_convert(
                op_output=output,
                fake_convert_params=fc_params,
                fake_convert_name=fc_name,
                data_type=data_type,
            )
            for inp_node in target_inputs:
                inp_node.replace_source_output(fc.output(0))
        else:
            raise nncf.InternalError(f"Incorrect target point type {transform_type}")

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
            raise nncf.InternalError("Constant node was expected but could not find it.")

        const_value = np.reshape(const_value, const_node.data.shape)

        shared_memory = True
        if const_port.get_element_type() == ov.Type.bf16:
            # Shared memory does not work for BF16 precision
            shared_memory = False

        new_const_node = OVModelTransformer._create_constant(
            const_value,
            dtype=const_port.get_element_type(),
            name=const_node.get_friendly_name(),
            shared_memory=shared_memory,
        )
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
        outputs_type = ov.Type.f32
        transformation = transformations[-1]
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)

        params, results = [], []
        for input_name, input_port_id in transformation.input_ids:
            input_node = name_to_node_mapping[input_name]
            if input_name in [tensor.node.get_friendly_name() for tensor in model.inputs]:
                params.append(input_node)
                continue

            input_port = input_node.input(input_port_id)
            input_type = input_port.get_element_type()
            input_node_output = input_port.get_source_output()
            parameter_name = get_parameter_node_name(input_name, input_port_id)

            new_param = opset.parameter(
                shape=input_node_output.partial_shape,
                dtype=outputs_type,
                name=parameter_name,
            )
            new_input = new_param.output(0)

            if input_type != outputs_type:
                new_input = opset.convert(new_param, destination_type=input_type).output(0)

            input_port.replace_source_output(new_input)
            new_param_tensors = [o.get_tensor() for o in new_param.outputs()]
            OVModelTransformer._update_tensor_names(new_param_tensors, [parameter_name])
            params.append(new_param)

        for output_name, output_port_id in transformation.output_ids:
            output_node = name_to_node_mapping[output_name]

            result_name = get_result_node_name(output_name, output_port_id)
            output_port = output_node.output(output_port_id)
            if output_port.get_element_type() != outputs_type:
                output_port = opset.convert(output_node, destination_type=outputs_type).output(0)
            new_result = opset.result(output_port, name=result_name)
            result_tensor_names = [result_name] + list(output_port.get_names())
            OVModelTransformer._update_tensor_names([new_result.get_output_tensor(0)], result_tensor_names)
            results.append(new_result)

        if not results:
            results = model.get_results()

        extracted_model = ov.Model(results, params)
        copy_rt_info(model, extracted_model, path=["nncf"])
        return extracted_model

    @staticmethod
    def _apply_stateless_model_extraction_transformation(
        model: ov.Model, transformations: List[OVStateLessModelExtractionCommand]
    ) -> ov.Model:
        """
        Extracts stateless sub-model from the original based on the inputs and outputs names.

        :param model: Model to apply transformations.
        :param transformation: Model extraction transformation.
        :return: Extracted sub-model.
        """
        transformation = transformations[-1]
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)

        params, results = [], []
        model_input_names = [tensor.node.get_friendly_name() for tensor in model.inputs]
        for input_name, output_port_id in transformation.input_ids:
            input_node = name_to_node_mapping[input_name]
            if input_name in model_input_names:
                params.append(input_node)
                continue

            output_port = input_node.output(output_port_id)
            parameter_name = get_parameter_node_name(input_name, output_port_id)
            new_param = opset.parameter(
                shape=output_port.partial_shape,
                dtype=output_port.get_element_type(),
                name=parameter_name,
            )
            for input_port in output_port.get_target_inputs():
                input_port.replace_source_output(new_param.output(0))
            new_param_tensors = [o.get_tensor() for o in new_param.outputs()]
            OVModelTransformer._update_tensor_names(new_param_tensors, [parameter_name])
            params.append(new_param)

        for output_name, output_port_id in transformation.output_ids:
            output_node = name_to_node_mapping[output_name]

            output_port = output_node.output(output_port_id)
            result_name = get_result_node_name(output_name, output_port_id)
            new_result = opset.result(output_port, name=result_name)
            OVModelTransformer._update_tensor_names([new_result.get_output_tensor(0)], [result_name])
            results.append(new_result)

        if not results:
            results = model.get_results()

        extracted_model = ov.Model(results=results, parameters=params)
        copy_rt_info(model, extracted_model, path=["nncf"])

        for op in extracted_model.get_ops():
            if op.get_type_name() != "ReadValue":
                continue

            op_input_values = op.input_values()
            op_outputs = op.outputs()
            if op_input_values:
                for op_output in op_outputs:
                    for target_in in op_output.get_target_inputs():
                        target_in.replace_source_output(op_input_values[0])
            else:
                raise RuntimeError("ReadValue has no initial value.")

        return extracted_model

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
        output_dtype = transformation.output_dtype
        if transform_type == TargetType.POST_LAYER_OPERATION:
            new_node = transformation.inplace_op_fn(target_node, port_id, transformation.last_inplace_node_name)
            return (new_node.output(fn_output_port_id), fn_output_port_id, output_dtype)
        if transform_type in [TargetType.PRE_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]:
            output = target_node.input_value(port_id)
            new_node = transformation.inplace_op_fn(
                output.get_node(), output.get_index(), transformation.last_inplace_node_name
            )
            return (new_node.output(fn_output_port_id), fn_output_port_id, output_dtype)
        raise nncf.InternalError(f"Transform type {transform_type} is not supported")

    @staticmethod
    def _apply_bias_insertion_transformations(
        model: ov.Model, transformations: List[OVBiasInsertionCommand]
    ) -> ov.Model:
        """
        Inserts bias operation after corresponding layer.

        :param transformations: List of the bias insertion transformations.
        :returns: Transformed model with null biases.
        """
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        for transformation in transformations:
            node_name = transformation.target_point.target_node_name
            node = name_to_node_mapping[node_name]
            # Since layers that may have biases mostly are Convolution or MatMul variations,
            # we may use only 0 output port.
            node_output_port = node.output(transformation.target_point.port_id)
            node_output_source_ports = node_output_port.get_target_inputs()

            bias_node_name = f"{node_name}/nncf_null_bias_"

            bias_const_node = OVModelTransformer._create_constant(
                transformation.bias_value,
                dtype=node_output_port.get_element_type(),
                name=f"{bias_node_name}/bias_value",
            )
            bias_const_output_port = bias_const_node.output(0)

            add_node = opset.add(node_output_port, bias_const_output_port, name=bias_node_name)

            for node_output_source_port in node_output_source_ports:
                node_output_source_port.replace_source_output(add_node.output(0))

        return model

    @staticmethod
    def _apply_multiply_insertion_transformations(
        model: ov.Model, transformations: List[OVMultiplyInsertionCommand]
    ) -> ov.Model:
        """
        Inserts Multiply with provided value for corresponding layer.

        :param transformations: List of the smooth insertion transformations.
        :returns: Transformed model with Multiply nodes.
        """
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)

        for transformation in transformations:
            node_name = transformation.target_point.target_node_name
            node = name_to_node_mapping[node_name]
            output_port_id = transformation.target_point.port_id
            node_output_port = node.output(output_port_id)

            destination_ports = []

            for target_input_port in node_output_port.get_target_inputs():
                target_node = target_input_port.get_node()
                if target_node.get_friendly_name() in transformation.destination_node_names:
                    destination_ports.append(target_input_port)

            scale_dtype = node_output_port.get_element_type()

            scale_constant = OVModelTransformer._create_constant(
                transformation.scale_value, dtype=scale_dtype, name=f"{transformation.multiply_node_name}/scale"
            )
            multiply_node = opset.multiply(node_output_port, scale_constant, name=transformation.multiply_node_name)

            for destination_port in destination_ports:
                destination_port.replace_source_output(multiply_node.output(0))

        return model

    @staticmethod
    def _apply_update_if_body_transformations(
        model: ov.Model, transformations: List[OVUpdateIfBodyCommand]
    ) -> ov.Model:
        """
        Update model body for IF node.

        :param model: Model to update and insert a new subgraph.
        :param transformations: Transformations with information of If node and an updated subgraph.
        :return: Original model with an updated subgraph.
        """
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        for transformation in transformations:
            subgraph_model = transformation.subgraph_model
            port_id = transformation.target_point.port_id
            node_name = transformation.target_point.target_node_name
            node = name_to_node_mapping[node_name]
            node.set_function(port_id, subgraph_model)
        return model

    @staticmethod
    def _apply_extract_if_body_transformation(
        model: ov.Model, transformations: List[OVExtractIfBodyCommand]
    ) -> ov.Model:
        """
        Extract a model body from If node.

        :param model: Model from which extracts a subgraph.
        :param transformations: Transformations with information from which
        If node and input port extract a model subgraph.
        :return: Model subgraph.
        """
        transformation = transformations[-1]
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        ov_node = name_to_node_mapping[transformation.if_node_name]
        if transformation.if_body_condition:
            return ov_node.get_function(0)
        return ov_node.get_function(1)
