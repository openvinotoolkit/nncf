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

from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter


class NNCFGraphFactory:
    @staticmethod
    def create(model):
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            return GraphConverter.create_nncf_graph(model)
        raise RuntimeError('Cannot create backend-specific graph'
                           'because {} is not supported!'.format(model_backend))
