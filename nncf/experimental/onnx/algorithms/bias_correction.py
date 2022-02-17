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

import onnx
import tempfile
import numpy as np

from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs

from nncf.experimental.post_training.statistics.statistics_collector import LayerStatistic
from nncf.experimental.post_training.algorithms.bias_correction import BiasCorrectionAlgorithm
from nncf.experimental.post_training.algorithms.bias_correction import BiasCorrectionAlgorithmParameters
from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph
from nncf.experimental.onnx.graph.transformations.commands import ONNXUpdateBias
from nncf.experimental.onnx.sampler import create_onnx_sampler

OPERATIONS_WITH_BIAS = ['BatchNormalization', 'Conv', ]


class ONNXBiasCorrectionAlgorithm(BiasCorrectionAlgorithm):
    def __init__(self, compressed_model: CompressedModel, engine, parameters: BiasCorrectionAlgorithmParameters):
        self.compressed_model = compressed_model
        self.engine = engine
        self.parameters = parameters

    def apply(self, model, layers_statistics, model_transformer):
        from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
        transformation_layout = ONNXTransformationLayout()

        original_model = self.compressed_model.original_model
        quantized_model = self.compressed_model.compressed_model
        layers_with_biases = self._get_layers_with_biases()

        layers_bias_shifts = {}
        for layer in layers_with_biases:
            bias_shift = self._get_output_of_layer(layer, original_model, quantized_model)
            layers_bias_shifts[layer] = bias_shift
            command = ONNXUpdateBias(layer.layer_name, bias_shift)
            transformation_layout.register(command)

            model_transformer._apply_transformation(command)
            onnx.save(self.compressed_model.compressed_model, '/home/aleksei/tmp/onnx/onnx_ptq_api/test.onnx')

        return model

    def get_layers_for_statistics(self, *args) -> List[LayerStatistic]:
        return self._get_outputs_with_biases()

    def get_transformation_commands(self, a):
        pass

    def _get_outputs_with_biases(self):
        output = []
        original_model_onnx_graph = ONNXGraph(self.compressed_model.original_model)

        for node in original_model_onnx_graph.get_all_nodes():
            if node.op_type in OPERATIONS_WITH_BIAS:
                outputs = original_model_onnx_graph.get_node_edges(node.name)['output'][0]
                layer_statistics = LayerStatistic(outputs)
                output.append(layer_statistics)
        return output

    def _get_layers_with_biases(self):
        output = []
        original_model_onnx_graph = ONNXGraph(self.compressed_model.original_model)

        for node in original_model_onnx_graph.get_all_nodes():
            if node.op_type in OPERATIONS_WITH_BIAS:
                layer_statistics = LayerStatistic(node.name)
                output.append(layer_statistics)
        return output

    def _get_output_of_layer(self, layer_name, model, quantized_model):
        sampler = create_onnx_sampler(self.engine)
        original_model_onnx_graph = ONNXGraph(self.compressed_model.original_model)
        layer_output = original_model_onnx_graph.get_node_edges(layer_name.layer_name)['output'][0]

        for i, sample in enumerate(sampler):
            _input, target = sample

            model_output = list(enumerate_model_node_outputs(model))[-1]
            model_with_intermediate_outputs = select_model_inputs_outputs(model,
                                                                          outputs=[layer_output, model_output])

            print(f'layer_output = {layer_output}')

            with tempfile.NamedTemporaryFile() as temporary_model:
                onnx.save(model_with_intermediate_outputs, temporary_model.name)
                self.engine.set_model(temporary_model.name)
                output = self.engine.infer(_input)
                layer_output_from_model = output[layer_output]

            model_output = list(enumerate_model_node_outputs(quantized_model))[-1]
            model_with_intermediate_outputs = select_model_inputs_outputs(quantized_model,
                                                                          outputs=[layer_output, model_output])

            with tempfile.NamedTemporaryFile() as temporary_model:
                onnx.save(model_with_intermediate_outputs, temporary_model.name)
                self.engine.set_model(temporary_model.name)
                output = self.engine.infer(_input)
                quantized_layer_output = output[layer_output]

            print(layer_output_from_model.shape)
            print(quantized_layer_output.shape)
            return np.mean(layer_output_from_model - quantized_layer_output, axis=(0, 2, 3))
