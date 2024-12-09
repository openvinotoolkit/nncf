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


from typing import Any, Dict, Optional, TypeVar

from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph

TModel = TypeVar("TModel")


class StateAttributes:
    """
    The state attributes.
    """

    EXAMPLE_INPUT_ARGS = "example_input_args"
    EXAMPLE_INPUT_KWARGS = "example_input_kwargs"


class ModelWrapper:
    """
    A wrapper class for the original model.

    :param _model: The original model to be wrapped.
    :param _graph: The graph representation of the model.
    :param state: The storage of the model state.
    """

    def __init__(
        self, model: TModel, graph: Optional[NNCFGraph] = None, state: Optional[Dict[str, Any]] = None
    ) -> None:
        self._model = model
        self._graph = graph
        self.state = state if state is not None else {}

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
                model=self.model,
                input_args=self.state.get(StateAttributes.EXAMPLE_INPUT_ARGS),
                input_kwargs=self.state.get(StateAttributes.EXAMPLE_INPUT_KWARGS),
            )
        return self._graph
