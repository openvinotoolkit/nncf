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

from abc import abstractmethod
from typing import TypeVar

from nncf.common.graph.graph import NNCFGraph
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup

TModel = TypeVar("TModel")


class NNCFQuantizer:
    @abstractmethod
    def get_quantization_setup(self, model: TModel, nncf_graph: NNCFGraph) -> SingleConfigQuantizerSetup:
        """
        Builds SingleConfigQuantizerSetup for the given model.

        :param model: Backend-specific model, for which Quantization Target Points are being seek.
        :param nncf_graph: NNCFGraph instance.
        :return: SingleConfigQuantizerSetup for the given model.
        """