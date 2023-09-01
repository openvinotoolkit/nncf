from collections import deque

import openvino.runtime as ov
from openvino.runtime import opset9 as opset
from tqdm import tqdm

from nncf import Dataset
from nncf.quantization.algorithms.post_training.backend import PostTrainingBackend


def make_transform_fn(input_descriptions):
    def transform_fn(data_item):
        inputs = []
        for desc in input_descriptions:
            inputs.append(data_item[desc.input_index])
        return tuple(inputs)

    return transform_fn


class OVPostTrainingBackend(PostTrainingBackend):
    def set_subgraph(self, subgraph_model, if_op, if_op_subgraph_port_id):
        if_op.set_function(if_op_subgraph_port_id, subgraph_model)

    def dump_model(self, model, dir, if_op, if_op_subgraph_port_id):
        name = if_op.get_friendly_name()
        if if_op_subgraph_port_id == 0:
            postfix = "then"
        if if_op_subgraph_port_id == 1:
            postfix = "else"
        model_path = f"{dir}/{name}_{postfix}.xml"
        ov.serialize(model, model_path)

    def is_single_model(self, model):
        for op in model.get_ops():
            if op.get_type_name() == "If":
                return False
        return True

    def add_results(self, model, node):
        results = model.get_results()
        params = model.get_parameters()
        assign_ops = [op for op in model.get_ops() if op.get_type_name() == "Assign"]

        extra_model_outputs = []
        result_names = []
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
            result_names.append(result_name)
        return (
            ov.compile_model(
                ov.Model(
                    results=results + extra_model_outputs, sinks=assign_ops, parameters=params, name=model.friendly_name
                )
            ),
            result_names,
        )

    def collect_dataset(self, model, ifnode, calibration_dataset, subset_size):
        dataset = []

        model_with_additional_results, result_names = self.add_results(model, ifnode)
        for inputs in tqdm(calibration_dataset.get_inference_data(list(range(subset_size))), total=subset_size):
            results = model_with_additional_results(inputs)
            data_item = []
            for name in result_names:
                data_item.append(results[name])
            dataset.append(tuple(data_item))
        return dataset

    def make_tasks(self, model, calibration_dataset, subset_size):
        tasks = deque()

        for if_op in [op for op in model.get_ops() if op.get_type_name() == "If"]:
            print(f"Collect dataset for If {if_op.get_friendly_name()} bodies:")
            datas = self.collect_dataset(model, if_op, calibration_dataset, subset_size)
            input_name = if_op.get_input_descriptions(0)
            transform_fn = make_transform_fn(input_name)

            tasks.append(
                (if_op.get_function(0), Dataset(datas, transform_fn), {"if_op": if_op, "if_op_subgraph_port_id": 0})
            )
            input_name = if_op.get_input_descriptions(1)
            transform_fn = make_transform_fn(input_name)
            tasks.append(
                (if_op.get_function(1), Dataset(datas, transform_fn), {"if_op": if_op, "if_op_subgraph_port_id": 1})
            )

        return tasks
