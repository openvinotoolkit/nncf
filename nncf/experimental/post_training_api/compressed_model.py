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

from abc import ABC
from abc import abstractmethod

from typing import TypeVar

from nncf.common.graph.graph import NNCFGraph

ModelType = TypeVar('ModelType')


class CompressedModel(ABC):
    """
    The original model wrapper used to build NNCFGraph and utilized it in the compression algorithms.
    """

    def __init__(self, model: ModelType):
        self.original_model = model
        self.nncf_graph = self.build_nncf_graph()
        self.transformations = []

    @abstractmethod
    def build_nncf_graph(self) -> NNCFGraph:
        """
        Builds NNCFGraph from the model.
        """
