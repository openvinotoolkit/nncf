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


from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TypeVar

from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend

TModel = TypeVar("TModel")


@dataclass
class ModelAttributes:
    """
    A class to store model attributes.

    :param example_input_args: Example input arguments for the model.
    :param example_input_kwargs: Example input keyword arguments for the model.
    """

    example_input_args: Optional[Tuple[Any]] = None
    example_input_kwargs: Optional[Dict[str, Any]] = None


class ModelWrapper:
    """
    A wrapper class for the original model.

    :param _model: The original model to be wrapped.
    :param _graph: The graph representation of the model.
    :param _attributes: The storage of the model attributes.
    :param _backend: The backend of the model.
    """

    def __init__(
        self, model: TModel, *, graph: Optional[NNCFGraph] = None, attributes: Optional[ModelAttributes] = None
    ) -> None:
        self._model = model
        self._graph = graph
        self._attributes = attributes or ModelAttributes()
        self._backend = get_backend(model)

    @property
    def model(self) -> TModel:
        """
        Retrieves the original model.
        """
        return self._model

    @property
    def graph(self) -> NNCFGraph:
        """
        Returns the NNCFGraph representation of the model.

        If the graph has not been created yet, it will be created using the model,
        example input arguments, and example input keyword arguments stored in the state.
        """
        if self._graph is None:
            self._graph = NNCFGraphFactory.create(
                self.model, self.attributes.example_input_args, self.attributes.example_input_kwargs
            )
        return self._graph

    @property
    def attributes(self) -> ModelAttributes:
        """
        Retrieves the model attributes.
        """
        return self._attributes

    @property
    def backend(self) -> BackendType:
        """
        Retrieves the model backend.
        """
        return self._backend

    def unwrap(self) -> Tuple[TModel, NNCFGraph]:
        """
        Retrieves the model and graph.

        :return: A tuple of the model and graph.
        """
        return self.model, self.graph
