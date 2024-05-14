# Copyright (c) 2024 Intel Corporation
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
from typing import List, Tuple

import openvino as ov

from nncf import Dataset
from nncf.common.factory import EngineFactory
from nncf.common.factory import ModelTransformerFactory
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.openvino.graph.metatypes.openvino_metatypes import OVLoopMetatype
from nncf.openvino.graph.node_utils import get_number_loop_op
from nncf.openvino.graph.node_utils import get_result_node_name
from nncf.openvino.graph.transformations.commands import OVOutputInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.quantization.algorithms.algorithm import Algorithm


def insert_outputs_before_loop_op(
    model: ov.Model,
    loop_node: NNCFNode,
) -> Tuple[ov.Model, List[str]]:
    """
    Insert Result nodes before Loop operation.

    :param model:
    :param loop_node:
    :return:
    """
    friendly_name_to_node = {op.get_friendly_name(): op for op in model.get_ops()}
    node = friendly_name_to_node[loop_node.node_name]

    body_parameter_index_to_output_name = {}
    transformation_layout = TransformationLayout()

    for input_desc in node.get_input_descriptions():
        name = node.input(input_desc.input_index).get_source_output().get_node().get_friendly_name()
        output_name = get_result_node_name(name, 0)
        body_parameter_index_to_output_name[input_desc.body_parameter_index] = output_name
        transformation_layout.register(
            OVOutputInsertionCommand(
                OVTargetPoint(TargetType.PRE_LAYER_OPERATION, loop_node.node_name, input_desc.input_index)
            )
        )

    model_transformer = ModelTransformerFactory.create(model)
    transformed_model = model_transformer.transform(transformation_layout)

    output_names = [
        output_name for _, output_name in sorted(body_parameter_index_to_output_name.items(), key=lambda x: x[0])
    ]

    return transformed_model, output_names


def create_calibration_dataset_for_loop_body(
    model: ov.Model, calibration_dataset: Dataset, subset_size: int, loop_node: NNCFNode
) -> Dataset:
    """
    :param model:
    :param calibration_dataset:
    :param loop_node:
    :return:
    """
    transformed_model, output_names = insert_outputs_before_loop_op(model, loop_node)

    data_items = []
    engine = EngineFactory.create(transformed_model)

    calibration_dataset_size = (
        min(subset_size, calibration_dataset.get_length())
        if calibration_dataset.get_length() is not None
        else subset_size
    )
    for input_data in track(
        islice(calibration_dataset.get_inference_data(), calibration_dataset_size),
        total=calibration_dataset_size,
        description="Create dataset for loop body",
    ):
        results = engine.infer(input_data)
        data_items.append([results[name] for name in output_names])

    return Dataset(data_items)


def get_loop_body(model: ov.Model, loop_node: NNCFNode) -> ov.Model:
    friendly_name_to_node = {op.get_friendly_name(): op for op in model.get_ops()}
    node = friendly_name_to_node[loop_node.node_name]
    return node.get_function()


def set_loop_body(model: ov.Model, loop_node: NNCFNode, body: ov.Model) -> ov.Model:
    friendly_name_to_node = {op.get_friendly_name(): op for op in model.get_ops()}
    node = friendly_name_to_node[loop_node.node_name]
    node.set_function(body)
    return model


def apply_algorithm_loop_bodies(
    algorithm: Algorithm,
    parent_model: ov.Model,
    parent_graph: NNCFGraph,
    parent_dataset: Dataset,
    subset_size: int,
    current_model_num: int,
    all_models_num: int,
) -> Tuple[ov.Model, int]:
    """
    Applies an algorithm recursievley to each body of Loop node.

    :param algorithm: Algorithm to apply.
    :param parent_model: Model to apply algorithm.
    :param parent_graph: Graph of a model.
    :param parent_dataset: Dataset for algorithm.
    :param subset_size: Size of a dataset to use for calibration.
    :param current_model_num: Current model number.
    :param all_models_num: All model numbers.
    :return: A model for every body of Loop nodes the algorithm was applied and the latest model number.
    """
    nncf_logger.info(f"Iteration [{current_model_num}/{all_models_num}] ...")
    quantized_model = algorithm.apply(parent_model, parent_graph, None, parent_dataset)

    if get_number_loop_op(parent_model) == 0:
        return quantized_model, current_model_num

    for loop_node in parent_graph.get_nodes_by_metatypes([OVLoopMetatype]):
        calibration_dataset = create_calibration_dataset_for_loop_body(
            parent_model, parent_dataset, subset_size, loop_node
        )
        body = get_loop_body(parent_model, loop_node)
        quantized_body, current_model_num = apply_algorithm_loop_bodies(
            algorithm,
            body,
            NNCFGraphFactory.create(body),
            calibration_dataset,
            subset_size,
            current_model_num + 1,
            all_models_num,
        )
        quantized_model = set_loop_body(quantized_model, loop_node, quantized_body)

    return quantized_model
