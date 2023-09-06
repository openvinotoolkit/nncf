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

from collections import deque
from itertools import islice
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Tuple

import openvino.runtime as ov
from openvino.runtime import opset9 as opset
from tqdm import tqdm

from nncf import Dataset
from nncf.data.dataset import DataItem
from nncf.quantization.algorithms.post_training.backend import PostTrainingBackend


def make_transform_fn(input_descriptions):
    def transform_fn(data_item):
        inputs = []
        for desc in input_descriptions:
            inputs.append(data_item[desc.input_index])
        return tuple(inputs)

    return transform_fn


class OVPostTrainingBackend(PostTrainingBackend):
    def _add_results(self, model: ov.Model, node: ov.Node) -> ov.Model:
        extra_model_outputs = []
        for input in node.inputs():
            output = input.get_source_output()
            output_name = output.get_node().get_friendly_name()
            result_name = f"{output_name}/if_output"
            result = opset.result(output, name=result_name)

            tensor = result.get_output_tensor(0)
            current_names = tensor.get_names()
            current_names.add(result_name)
            tensor.set_names(current_names)

            extra_model_outputs.append(result)
        return ov.Model(
            results=extra_model_outputs,
            sinks=[op for op in model.get_ops() if op.get_type_name() == "Assign"],
            parameters=model.get_parameters(),
            name=model.friendly_name,
        )

    def _collect_dataset(
        self, model: ov.Model, if_op: ov.Node, calibration_dataset: Dataset, subset_size: int
    ) -> Iterable[DataItem]:
        dataset = []
        model_with_additional_results = self._add_results(model, if_op)
        compiled_model = ov.compile_model(model_with_additional_results)
        for input_data in tqdm(
            islice(calibration_dataset.get_inference_data(), subset_size),
            total=subset_size,
            desc=f"Collect dataset for If {if_op.get_friendly_name()} operation:",
        ):
            results = compiled_model(input_data)
            # TODO: Use only inputs which are passed to subgraph infers. E.g. in example 0-index is never used.
            dataset.append(tuple(results.values()))
        return dataset

    def _make_task(
        self, if_op: ov.Node, port_id: int, dataset: Iterable[DataItem]
    ) -> Tuple[ov.Model, Dataset, Dict[str, Any]]:
        assert if_op.get_type_name() == "If"
        input_name = if_op.get_input_descriptions(port_id)
        transform_fn = make_transform_fn(input_name)
        return (
            if_op.get_function(port_id),
            Dataset(dataset, transform_fn),
            {"if_op": if_op, "if_op_subgraph_port_id": port_id},
        )

    def set_subgraph(self, subgraph_model: ov.Model, if_op: ov.Node, if_op_subgraph_port_id: int) -> None:
        if_op.set_function(if_op_subgraph_port_id, subgraph_model)

    def dump_model(self, model: ov.Model, dir: str, if_op: ov.Node, if_op_subgraph_port_id: int) -> None:
        name = if_op.get_friendly_name().replace("/", "")
        if if_op_subgraph_port_id == 0:
            postfix = "then"
        if if_op_subgraph_port_id == 1:
            postfix = "else"
        model_name = f"{name}_{postfix}.xml"
        model_path = Path(dir) / model_name
        ov.serialize(model, model_path)

    def is_single_model(self, model: ov.Model) -> bool:
        for op in model.get_ops():
            if op.get_type_name() == "If":
                return False
        return True

    def make_tasks(
        self, model: ov.Model, calibration_dataset: Dataset, subset_size: int
    ) -> List[Tuple[ov.Model, Dataset, Dict[str, Any]]]:
        tasks = []
        for op in model.get_ops():
            if op.get_type_name() == "If":
                subgraph_dataset = self._collect_dataset(model, op, calibration_dataset, subset_size)
                for port_id in (0, 1):
                    tasks.append(self._make_task(op, port_id, subgraph_dataset))
        return tasks
