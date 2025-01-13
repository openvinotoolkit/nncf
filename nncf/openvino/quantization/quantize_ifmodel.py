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

from itertools import islice
from typing import Dict, List, Optional, Tuple

import openvino.runtime as ov

from nncf import Dataset
from nncf.common import factory
from nncf.common.engine import Engine
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.openvino.graph.metatypes.openvino_metatypes import OVIfMetatype
from nncf.openvino.graph.model_utils import remove_friendly_name_duplicates
from nncf.openvino.graph.node_utils import get_number_if_op
from nncf.openvino.graph.transformations.commands import OVExtractIfBodyCommand
from nncf.openvino.graph.transformations.commands import OVOutputInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import OVUpdateIfBodyCommand
from nncf.quantization.algorithms.algorithm import Algorithm


def _make_dataset_for_if_bodies(
    engine: Engine,
    calibration_dataset: Dataset,
    if_cond_input_name: str,
    then_model_input_names: List[str],
    else_model_input_names: List[str],
    subset_size: int,
) -> Tuple[Dataset, Dataset]:
    """
    Returns dataset for a then and else bodies of If node.

    :param engine: Engine to infer parent model to obtain dataitems for a child dataset.
    :param calibration_dataset: Dataset to infer parent model.
    :param if_cond_input_name: Input name of If node condition.
    :param then_model_input_names: Names of inputs for then body
    (should be in the order of passing them to a model).
    :param else_model_input_names: Names of inputs for else body
    (should be in the order of passing them to a model).
    :param subset_size: The size of calibration_dataset.
    :return Dataset: Dataset for child model.
    """

    then_dataset, else_dataset = [], []
    calibration_dataset_size = (
        min(subset_size, calibration_dataset.get_length())
        if calibration_dataset.get_length() is not None
        else subset_size
    )
    for input_data in track(
        islice(calibration_dataset.get_inference_data(), calibration_dataset_size),
        total=calibration_dataset_size,
        description="Collecting the dataset for then and else bodies:",
    ):
        data_item = []
        results = engine.infer(input_data)
        if results[if_cond_input_name]:
            for name in then_model_input_names:
                data_item.append(results[name])
            then_dataset.append(data_item)
        else:
            for name in else_model_input_names:
                data_item.append(results[name])
            else_dataset.append(data_item)
    nncf_logger.info(f"The length of dataset for then body is {len(then_dataset)}, else body is {len(else_dataset)}.")
    return Dataset(then_dataset), Dataset(else_dataset)


def _extract_if_body(model_transformer: ModelTransformer, if_node: NNCFNode, if_body_condition: bool) -> ov.Model:
    """
    Returns if body of If node based on a value of if_body_condition.

    :param model_transformer: ModelTransformer instance.
    :param if_node: If node.
    :param if_submodel_condition: If True returns then body of If node, otherwise - else body.
    :return: If body.
    """
    transformation_layout = TransformationLayout()
    command = OVBackend.create_extract_if_body_command(if_node, if_body_condition)
    transformation_layout.register(command)
    return model_transformer.transform(transformation_layout)


def _update_if_body(
    model_transformer: ModelTransformer, if_node: NNCFNode, if_body_condition: bool, body: ov.Model
) -> ov.Model:
    """
    Update body of If node, based on if_body_condition.

    :param model_transformer: ModelTransformer instance.
    :param if_node: If node.
    :param if_body_condition: Condition of If node body.
    :param body: New body.
    :return: Updated model with a new body of If node.
    """
    transformation_layout = TransformationLayout()
    command = OVBackend.create_update_body_command(if_node, if_body_condition, body)
    transformation_layout.register(command)
    return model_transformer.transform(transformation_layout)


def _add_outputs_before_if_node(model_transformer: ModelTransformer, model: ov.Model, if_node: NNCFNode) -> ov.Model:
    """
    Inserts extra outputs on If node inputs.

    :param model_transformer: ModelTransformer instance.
    :param model: Model instance.
    :param if_node: If node.
    :return: Model with extra outputs before If node.
    """
    transformation_layout = TransformationLayout()
    for command in OVBackend.create_output_insertion_commands_if_node(model, if_node):
        transformation_layout.register(command)
    return model_transformer.transform(transformation_layout)


def apply_algorithm_if_bodies(
    algorithm: Algorithm,
    parent_model: ov.Model,
    graphs: Dict[str, NNCFGraph],
    graph_id: str,
    parent_dataset: Dataset,
    subset_size: int,
    current_model_num: int,
    parent_statistic_points: Optional[StatisticPointsContainer] = None,
) -> Tuple[ov.Model, int]:
    """
    Applies an algorithm recursively to each bodies of If node.

    :param parent_model: Model to apply algorithm.
    :param graphs: All model graphs.
    :param graph_id: Current graph id in the graphs.
    :param parent_dataset: Dataset for algorithm.
    :param subset_size: Size of a dataset to use for calibration.
    :param current_model_num: Current model number.
    :param parent_statistic_points: Statistics points for algorithm.
    :return: A model for every bodies of If nodes the algorithm was applied and the latest model number.
    """
    nncf_logger.info(f"Iteration [{current_model_num}/{len(graphs)}] ...")
    parent_graph = graphs[graph_id]
    quantized_model = algorithm.apply(parent_model, parent_graph, parent_statistic_points, parent_dataset)
    if get_number_if_op(parent_model) == 0:
        return quantized_model, current_model_num
    model_transformer_fp32 = factory.ModelTransformerFactory.create(parent_model)
    for if_node in parent_graph.get_nodes_by_metatypes(OVBackend.if_node_metatypes()):
        parent_model_with_additional_outputs = _add_outputs_before_if_node(
            model_transformer_fp32, parent_model, if_node
        )
        then_model_input_names = OVBackend.get_if_body_input_names(parent_model, if_node, True)
        else_model_input_names = OVBackend.get_if_body_input_names(parent_model, if_node, False)
        if_cond_input_name = OVBackend.get_if_cond_input_name(parent_model_with_additional_outputs, if_node)
        then_dataset, else_dataset = _make_dataset_for_if_bodies(
            factory.EngineFactory.create(parent_model_with_additional_outputs),
            parent_dataset,
            if_cond_input_name,
            then_model_input_names,
            else_model_input_names,
            subset_size,
        )

        then_model = _extract_if_body(model_transformer_fp32, if_node, True)
        then_model = remove_friendly_name_duplicates(then_model)
        else_model = _extract_if_body(model_transformer_fp32, if_node, False)
        else_model = remove_friendly_name_duplicates(else_model)

        then_quantized_model, current_model_num = apply_algorithm_if_bodies(
            algorithm,
            then_model,
            graphs,
            if_node.node_name + "_then",
            then_dataset,
            subset_size,
            current_model_num + 1,
        )
        else_quantized_model, current_model_num = apply_algorithm_if_bodies(
            algorithm,
            else_model,
            graphs,
            if_node.node_name + "_else",
            else_dataset,
            subset_size,
            current_model_num + 1,
        )
        model_transformer_int8 = factory.ModelTransformerFactory.create(quantized_model)
        quantized_model = _update_if_body(model_transformer_int8, if_node, True, then_quantized_model)
        model_transformer_int8 = factory.ModelTransformerFactory.create(quantized_model)
        quantized_model = _update_if_body(model_transformer_int8, if_node, False, else_quantized_model)
    return quantized_model, current_model_num


class OVBackend:
    @staticmethod
    def _get_if_body_port_id(if_body_condition: bool):
        """
        Returns port id of a If body based on if_body_condition.

        :param if_body_condition: Condition of If node.
        :return: Port id of body of If node.
        """
        return int(not if_body_condition)

    @staticmethod
    def if_node_metatypes() -> List[OperatorMetatype]:
        """
        Returns metatypes that map to If node.

        :return: Metatypes mapped to If node.
        """
        return [OVIfMetatype]

    @staticmethod
    def get_if_body_input_names(model: ov.Model, if_node: NNCFNode, if_body_condition: bool) -> List[str]:
        """
        Returns input names of If node body based on if_body_condition.
        The order of inputs are in a way that they are passed to the model during inference.

        :param model: Original model.
        :param if_node: If node.
        :param if_body_condition: True for then body, else for else body.
        :return: Input names of If body.
        """
        input_names = []
        name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        ov_node = name_to_node_mapping[if_node.node_name]
        input_indices = [
            desc.input_index
            for desc in ov_node.get_input_descriptions(OVBackend._get_if_body_port_id(if_body_condition))
        ]
        for index in input_indices:
            input_names.append(ov_node.inputs()[index].get_tensor().get_any_name())
        return input_names

    @staticmethod
    def get_if_cond_input_name(model: ov.Model, if_node: NNCFNode) -> str:
        """
        Returns name of condition input of If node.

        :param model: Model.
        :param if_node: If node.
        :return: Name of condition input of If node.
        """
        name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        ov_node = name_to_node_mapping[if_node.node_name]
        return ov_node.inputs()[0].get_tensor().get_any_name()

    @staticmethod
    def create_update_body_command(if_node: NNCFNode, if_body_condition: bool, body: ov.Model) -> OVUpdateIfBodyCommand:
        """
        Returns a command for setting a body of If node by a new one.

        :param if_node: If node.
        :param if_body_condition: Condition of If node.
        :param body: A new body to set.
        :return: Command to update If node body.
        """
        target_point = OVTargetPoint(
            TargetType.LAYER, if_node.node_name, OVBackend._get_if_body_port_id(if_body_condition)
        )
        return OVUpdateIfBodyCommand(target_point, body)

    @staticmethod
    def create_extract_if_body_command(if_node: NNCFNode, if_body_condition: bool) -> OVExtractIfBodyCommand:
        """
        Returns a command for extraction body of If node.
        If if_body_condition is True, extract then body, otherwise - else body.

        :param if_node: If node.
        :param if_body_condition: Condition of body of If node.
        :return: Extracted body of If node.
        """
        return OVExtractIfBodyCommand(if_node.node_name, if_body_condition)

    @staticmethod
    def create_output_insertion_commands_if_node(model: ov.Model, if_node: NNCFNode) -> List[OVOutputInsertionCommand]:
        """
        Returns output insertion commands on If node inputs.

        :param ov.Model model: Model.
        :param NNCFNode if_node: If node.
        :return: Transformation commands to insert outputs before If node.
        """
        assert if_node.metatype == OVIfMetatype
        commands = []
        name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        ov_node = name_to_node_mapping[if_node.node_name]
        for port_id, ov_input in enumerate(ov_node.inputs()):
            target_point = OVTargetPoint(TargetType.PRE_LAYER_OPERATION, if_node.node_name, port_id)
            ov_input_dtype = ov_input.get_element_type()
            commands.append(OVOutputInsertionCommand(target_point, output_dtype=ov_input_dtype))
        return commands
