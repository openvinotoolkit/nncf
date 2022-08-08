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

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.definitions import NNCFGraphNodeType

from nncf.experimental.post_training.api.dataset import NNCFData
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.sampler import Sampler
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.onnx.samplers import create_onnx_sampler
from nncf.experimental.onnx.tensor import ONNXNNCFTensor
from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter

import onnxruntime as rt
import numpy as np
import onnx


class ONNXEngine(Engine):
    """
    Engine for ONNX backend using ONNXRuntime to infer the model.
    """

    def __init__(self, **rt_session_options):
        super().__init__()
        self._inputs_transforms = lambda input_data: input_data.astype(np.float32)
        self.sess = None
        self.input_names = set()
        self.rt_session_options = rt_session_options

        # TODO: Do not force it to use CPUExecutionProvider
        # OpenVINOExecutionProvider raises the following error.
        # onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1
        # : FAIL : This is an invalid model. Error: Duplicate definition of name (data).
        self.rt_session_options['providers'] = ['CPUExecutionProvider']

    def get_sampler(self) -> Sampler:
        # TODO (Nikita Malinin): Replace range calling with the max length variable
        return self.sampler if self.sampler else create_onnx_sampler(self.dataset, range(len(self.dataset)))

    def set_model(self, model: onnx.ModelProto) -> None:
        """
        Creates ONNXRuntime InferenceSession for the onnx model.

        :param model: onnx.ModelProto model instance
        """
        super().set_model(model)
        serialized_model = model.SerializeToString()
        self.sess = rt.InferenceSession(serialized_model, **self.rt_session_options)

        self.input_names.clear()
        for inp in self.sess.get_inputs():
            self.input_names.add(inp.name)

        self._create_model_graphs(model)

    def infer(self, input_data: NNCFData) -> NNCFData:
        """
        Runs model on the provided input_data via ONNXRuntime InferenceSession.
        Returns the dictionary of model outputs by node names.

        :param input_data: inputs for the model transformed with the inputs_transforms
        :return output_data: models output after outputs_transforms
        """
        output_tensors = self.sess.run(
            [], {k: v.tensor for k, v in input_data.items() if k in self.input_names})
        model_outputs = self.sess.get_outputs()

        return {
            output.name: ONNXNNCFTensor(tensor)
            for tensor, output in zip(output_tensors, model_outputs)
        }

    def _create_model_graphs(self, model):
        self.nncf_graph = GraphConverter.create_nncf_graph(model)
        self.onnx_graph = ONNXGraph(model)

    def _register_statistics(self, outputs: NNCFData, statistic_points: StatisticPointsContainer) -> None:
        edge_name_to_node_name = {}
        for node_name, _statistic_points in statistic_points.items():
            for statistic_point in _statistic_points:
                if NNCFGraphNodeType.INPUT_NODE in statistic_point.target_point.target_node_name:
                    nncf_node_name = self.nncf_graph.get_node_by_name(statistic_point.target_point.target_node_name)
                    onnx_nodes_after_input_node = [edge.to_node for edge in
                                                   self.nncf_graph.get_output_edges(nncf_node_name)]
                    for onnx_node_name in onnx_nodes_after_input_node:
                        edge_name = self.onnx_graph.get_node_edges(onnx_node_name.node_name)['input'][0]
                        edge_name_to_node_name[edge_name] = node_name
                elif statistic_point.target_point.type == TargetType.POST_LAYER_OPERATION:
                    edge_name = self.onnx_graph.get_node_edges(node_name)['output'][0]
                elif statistic_point.target_point.type == TargetType.PRE_LAYER_OPERATION:
                    edge_name = self.onnx_graph.get_node_edges(node_name)['input'][0]
                else:
                    RuntimeError('The statistics should be collected only from the input of output edges of the node')
                edge_name_to_node_name[edge_name] = node_name

        for output_name, output_tensor in outputs.items():
            if output_name in edge_name_to_node_name:
                node_name = edge_name_to_node_name[output_name]
                for statistic_point in statistic_points[node_name]:
                    statistic_point.register_tensor(output_tensor)
